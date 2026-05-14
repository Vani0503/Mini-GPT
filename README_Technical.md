# Mini GPT — Building a Language Model from Scratch

**A PM's attempt to stop treating LLMs as black boxes.**

---

## What This Is

This project documents my attempt to build a miniature GPT-style language model entirely from scratch using PyTorch — no Hugging Face, no pre-built pipelines, no API calls. I am a Product Manager and I built this not to produce a working chatbot, but to develop genuine intuition about how LLMs behave, why they fail, and what architectural choices drive product outcomes like hallucination, latency, cost, and trust.

The codebase lives across five files that track a 10-day learning sprint. Each file represents a distinct phase: tokenization, embeddings, attention, transformer architecture, and training. This README is meant as a revision artifact — something I can read 6 months from now to remember what I built, what decisions I made, and what surprised me.

---

## Why Build Instead of Using APIs

The short answer: APIs abstract everything in a way that makes it impossible to reason about failure. As a PM, I was routinely making decisions about context windows, hallucination risks, retrieval architectures, and prompt design — but my understanding was fundamentally superficial. I could describe what these things were. I could not reason about why they behaved the way they did.

Building from scratch forced me to encounter the actual mechanisms. I now understand why attention is expensive at long sequences, why temperature flattens probability distributions, why the same model produces worse outputs on out-of-distribution prompts, and why backpropagation through 96 layers requires significantly more compute than inference. These aren't abstract concepts anymore — I wrote the code and watched the numbers change.

---

## Project Structure

```
mini-gpt/
├── tokenizer.py               # Days 1–3: tokenization, embeddings, manual attention
├── day4_transformer.py        # Day 4: transformer block with PyTorch primitives
├── mini_gpt.py                # Days 5–6: full mini GPT architecture + training
├── day7_generation.py         # Day 7: positional encoding, causal masking, generation, decoding
├── training_corpus.txt        # personal writing used as training data
├── mini_gpt_checkpoint.pth    # Saved model weights + vocabulary
└── README.md
```

**Deployed:** A Streamlit app comparing Mini-GPT outputs vs ChatGPT responses side by side — same prompt, both models, observable differences. Mini-GPT was trained on a small corpus of personal writing; its outputs are an incoherent statistical remix of that training text. ChatGPT's outputs are coherent and grounded. The contrast makes distribution gaps and hallucination behavior concrete and visible.

---

## Day 1 — Tokenization

**What I built:** A character-level tokenizer from scratch.

```python
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(tokens): return "".join([itos[i] for i in tokens])
```

**What this means:** The tokenizer converts raw text into integer IDs — one per unique character. "hello world" with 8 unique characters produces `vocab_size = 8`. The model never sees raw strings again after this step.

**Decisions made:**
- Used character-level tokenization rather than word-level or subword (BPE). This keeps vocabulary tiny (~30–100 tokens) and makes every transformation visible. The tradeoff is that character-level models learn spelling patterns more than semantics — they don't naturally develop word-level meaning.
- Real GPT models use subword tokenization (BPE) with ~50,000+ vocabulary tokens, because subword units carry more semantic content per token and are more compute-efficient.

**PM insight — token count is a cost driver:** Every token in a prompt contributes to inference cost through two mechanisms. First, attention computes Q@K^T across all tokens in the context, which is O(T²) — doubling the number of tokens quadruples attention computation. Second, the feedforward network runs once per token position per layer. On the memory side, serving APIs typically maintain a KV cache storing key and value vectors for every token processed — more tokens means more GPU memory held per request. This is why long system prompts, verbose few-shot examples, or unbounded conversation histories make LLM products expensive at scale.

---

## Day 2 — Embeddings

**What I built:** A learnable lookup table mapping token IDs to continuous vectors.

```python
vocab_size = len(chars)
embed_dim = 64

embedding = nn.Embedding(vocab_size, embed_dim)
```

**What this means:** The embedding layer creates a matrix of shape `(vocab_size × embed_dim)`. Each token ID maps to one row — a 64-dimensional vector. These vectors start random and get updated during training. By end of training, semantically related tokens cluster together in vector space.

**Decisions made:**
- `embed_dim = 64` — chosen as a learning-friendly middle ground. Too small (e.g., 8) and the model lacks expressive capacity. Too large and outputs become unreadable during debugging. Real models use 768 (GPT-2 small) to 12,288 (GPT-3).
- Does not have to be a perfect square or power of 2. `embed_dim = 37` is technically valid. The convention toward powers of 2 is hardware optimization, not mathematical necessity.

**What surprised me:** The 64 numbers in each token vector are not interpretable human concepts like "positivity" or "animateness." Meaning exists in the geometric relationships between vectors — in the combination, not individual dimensions. The king–man+woman≈queen arithmetic works because of how training pressure shapes the entire vector space collectively.

**Key output shapes I tracked:**
- Input sentence "world" (5 chars) → `encoded = [7, 5, 6, 4, 1]` → `embedded.shape = (5, 64)`
- With batch dimension: `(1, 5, 64)` — batch, sequence length, embedding dim
- This `(B, T, C)` shape appears everywhere in transformers

**PM insight — three levels of embedding:** Token-level (GPT internally), sentence-level (search, deduplication), and document-level (RAG). Knowing which level you're working at matters for choosing the right architecture. Spotify recommendations use a different embedding strategy than a customer support bot.

---

## Day 3 — Attention

**What I built:** Causal self-attention from scratch, without any PyTorch attention primitives.

```python
W_Q = torch.randn(embed_dim, embed_dim)
W_K = torch.randn(embed_dim, embed_dim)
W_V = torch.randn(embed_dim, embed_dim)

Q = embedded @ W_Q   # (5, 64)
K = embedded @ W_K   # (5, 64)
V = embedded @ W_V   # (5, 64)

scores = Q @ K.T     # (5, 5)

d_k = embed_dim
scaled_scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

mask = torch.tril(torch.ones(5, 5))
scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))

attention_weights = F.softmax(scaled_scores, dim=-1)   # (5, 5)

output = attention_weights @ V   # (5, 64)
```

**What each piece does:**

*Query, Key, Value projections:* The same input embedding is linearly projected three times into three different representations. Query = "what am I looking for?" Key = "what information do I contain?" Value = "what information will I give you if selected?" These W matrices start random and get trained. The model discovers what meaningful Queries and Keys look like through gradient descent — nobody labels them.

*Attention scores `Q @ K.T`:* This produces a 5×5 matrix where entry `[i][j]` measures how much token i should attend to token j. Each entry is a dot product similarity across all 64 dimensions — not per-dimension scores, but a single scalar representing global relevance.

*Scaling by `√d_k`:* Dot products in high-dimensional spaces produce very large values. These push softmax into extreme distributions (one token gets ~100% attention, rest get ~0%). Dividing by `√64 = 8` keeps the values in a range where softmax produces meaningful gradients. This was a critical engineering insight from the original Attention Is All You Need paper.

*Causal masking:* Without masking, token 3 can see tokens 4 and 5 during training. That's cheating — when the model learns to predict token 4, it shouldn't have already seen it. `torch.tril` creates a lower-triangular mask. Positions set to zero get replaced by `-inf`, which softmax converts to 0 attention weight. Now each token only attends to previous tokens and itself.

*Output `attention_weights @ V`:* Each token's final representation becomes a weighted sum of all other tokens' Value vectors, weighted by how much attention was assigned. Token 'w' in "world" no longer represents just 'w' — it now contains information borrowed from 'o', 'r', 'l', 'd' proportional to their relevance.

**What surprised me most:** Attention is not "understanding language." It is relevance-weighted information sharing between tokens. The model doesn't "know" that "bank" refers to a river bank versus a financial bank — it learns attention patterns that statistically help next-token prediction, and those patterns happen to encode contextual disambiguation.

**Output I saw (attention weights for "world"):**
```
tensor([[4.2695e-30, 4.2353e-41, 1.0000e+00, 9.5629e-29, ...],
        [1.0000e+00, 7.6094e-40, 1.2046e-13, 7.7874e-29, ...],
        ...])
```
Each row sums to 1. With random weights, attention becomes extremely sharp (one token dominates). After training, distributions spread into more meaningful patterns.

**Confusion I resolved:** Early on I thought Q, K, V were computed per dimension. They're not. Q and K produce one scalar relevance score per token pair via dot product across all dimensions. That single score then scales the entire V vector.

**PM insight — one place hallucinations can start:** Wrong attention is one contributing mechanism to hallucination — if attention focuses on irrelevant tokens, the Value aggregation mixes in irrelevant information and the output is built on faulty context. But hallucination has multiple origins: the training data itself may have contained wrong or conflicting information; the model may never have seen the concept at all and the entire forward pass is extrapolating; or the sampling step may simply pick a low-probability token by chance. Wrong attention is a contributor, not the single root cause. RAG helps by providing grounded context that attention can correctly focus on, reducing the extrapolation problem.

---

## Day 4 — Transformer Block

**What I built:** A proper reusable transformer block using PyTorch's MultiheadAttention.

```python
class MiniTransformerBlock(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=2,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x
```

**Why each component exists:**

*Multi-head attention (`num_heads=2`):* Instead of one Query/Key/Value triplet, we run two in parallel. Each head learns different relationship types — one might specialize in local syntax, another in long-range dependencies. The model discovers this specialization automatically through training. Nobody labels what head 1 is "for." Researchers studying interpretability have found heads that track bracket matching in code, resolve pronouns, and identify subject-verb relationships.

*Residual connections (`x = norm(x + f(x))`):* Here x is the tensor of all token representations at that point in the network — shape (B, T, embed_dim), meaning all token vectors for the current batch stacked together. The formula `x = norm(x + f(x))` means: pass the current representations through a transformation f (either attention or feedforward), then add the result back elementwise to whatever x was before that transformation. The original information is preserved; f only contributes a delta. This was a breakthrough from ResNet (Kaiming He, ~2015). Without residuals, each layer overwrites the previous representation, gradients struggle to flow backward through many layers, and deep networks become unstable. With residuals, each layer contributes refinements on top of preserved earlier information.

*Feedforward network (expand–ReLU–compress):* Attention mixes information across tokens. The feedforward network transforms information within each token position independently — no information passes between positions here. In code, PyTorch runs it as a single matrix operation across the whole batch for efficiency, but the Linear layers operate on the last dimension (embed_dim), so each of the T token vectors is transformed separately. The expansion to `embed_dim × 4` before compression creates a wider intermediate space for richer nonlinear computation. ReLU (`max(0, x)`) introduces nonlinearity — without it, the entire network collapses into a single linear transformation regardless of depth.

*LayerNorm:* After attention and after feedforward, activations can grow large or shrink toward zero as they pass through many layers, making training unstable. LayerNorm fixes this by normalizing each token's feature vector independently: it computes the mean and variance across all embed_dim values for that token, then rescales — `(x - mean) / sqrt(variance + epsilon)`. It then applies two small learned parameters (scale and shift) on top. Result: each token vector entering the next operation has roughly zero mean and unit variance. Crucially, LayerNorm operates per-token across features — not across the batch — so it works regardless of batch size.

**Decisions made:**
- Used 2 stacked blocks in mini_gpt.py. Real GPT-2 small uses 12; GPT-3 uses 96. Each layer refines representations progressively — early layers tend to learn local syntax, middle layers sentence structure, deep layers abstract reasoning-like patterns. Nobody programs this hierarchy — it emerges from repeated optimization.
- Kept `embed_dim = 16` in the Day 4 file for visual interpretability, then moved back to 64 in the full model.

**Output shapes confirmed:**
```
Input shape: torch.Size([1, 5, 8])
Attention output: torch.Size([1, 5, 8])
After residual + norm: torch.Size([1, 5, 8])
Feedforward output: torch.Size([1, 5, 8])
Final transformer output: torch.Size([1, 5, 8])
```
Shape is preserved throughout. The transformer doesn't change how many tokens or how many dimensions there are — it changes the information inside the vectors.

**PM insight — why scaling improves reasoning:** No one explicitly programmed reasoning into GPT. Reasoning-like behavior emerged from stacking many transformer blocks and training at massive scale. Each layer creates slightly richer abstractions; enough layers creates the capacity for chain-of-thought-style computation. This was genuinely surprising to researchers when it appeared — it's why scaling law papers became so significant in AI strategy.

---

## Day 5 — Mini GPT Architecture

**What I built:** A full prediction pipeline moving from raw text to vocabulary probability distributions.

The key new addition was the output projection layer:

```python
linear = nn.Linear(embed_dim, vocab_size)
logits = linear(x)           # (1, 16, 9)
probs = F.softmax(logits, dim=-1)  # (1, 16, 9)
```

And the input/target pair construction:

```python
x = data[:-1]   # input: all tokens except last
y = data[1:]    # target: all tokens shifted by one
```

**What input/target pairs mean:** For the sentence "the cat sat on the mat", training creates these prediction tasks: given "the" → predict "cat"; given "the cat" → predict "sat"; and so on. This is the entire GPT training objective. The model never learns grammar, meaning, or facts explicitly — it learns statistical continuation patterns. Everything else emerges.

**The `(1, 16, 9)` output shape:** With vocabulary size 9, the final linear layer maps each token's 16-dimensional representation to 9 scores — one per possible next word. After softmax, these become probabilities. For every one of the 16 input positions, the model simultaneously produces a probability distribution over the entire vocabulary. This explains why large vocabularies (50k+ tokens in real GPT) are expensive: every position produces 50k scores per forward pass.

**Decisions made:**
- Switched from character-level to word-level tokenization for this phase, so the model learns something closer to semantic patterns.
- Used `embed_dim = 16` in the mini GPT (down from 64) to keep tensor shapes readable.

**PM insight — data quality shapes everything:** A model trained on 2 sentences learns almost nothing generalizable. Increasing to 50–100 structured sentences with diverse vocabulary and repeated grammatical patterns produces models that actually exhibit interesting generation behavior. The observation that data quality often matters more than model size in determining behavior quality is now foundational in AI strategy discussions.

---

## Day 6 — Training

**What I built:** A full training loop with cross-entropy loss, AdamW optimizer, and backpropagation.

```python
model = MiniGPT(vocab_size, embed_dim)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

for step in range(200):
    logits = model(x)
    logits = logits.view(-1, vocab_size)
    targets = y.view(-1)

    loss = loss_fn(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")
```

**What I observed:**
```
Step 0   | Loss: 2.3968
Step 20  | Loss: 1.7585
Step 40  | Loss: 1.2816
Step 60  | Loss: 1.0150
Step 80  | Loss: 0.8958
Step 100 | Loss: 0.8317
Step 120 | Loss: 0.7923
Step 140 | Loss: 0.7661
Step 160 | Loss: 0.7477
Step 180 | Loss: 0.7343
```

The model was genuinely learning. Early improvements are large because the model moves away from uniform random predictions quickly. Later improvements are smaller because easy patterns are already learned and harder statistical dependencies require more updates.

**What cross-entropy loss measures:** How much probability mass was assigned to the correct next token. If the target is "cat" and the model assigned probability 0.70 to "cat" and 0.30 distributed across everything else, loss is relatively low. If it assigned 0.02 to "cat," loss is high. Backpropagation traces how each weight contributed to that error and adjusts accordingly.

**What backpropagation actually does:** It uses the chain rule from calculus to efficiently compute how sensitive the final loss is to every single parameter — billions in a real model. PyTorch builds a computation graph during the forward pass and traverses it backward using `loss.backward()`. No experimental weight-by-weight testing: calculus gives us the gradient (direction and magnitude of change needed) for all parameters simultaneously. This is why GPUs are essential: billions of gradient calculations can be parallelized as matrix operations.

**Why training is much more expensive than inference:** The forward pass produces a prediction. The backward pass computes gradients through every layer — as expensive or more than forward. Then the optimizer updates all parameters. Training requires storing intermediate activations from the forward pass so backprop can use them. Inference only needs the forward pass and discards gradients. This is why training GPT-3 cost millions of dollars in compute while inference is comparatively cheap.

**Gradient descent vs backpropagation — the distinction:**
- Backpropagation: computes blame (how much did each weight contribute to the error?)
- Gradient descent: uses that blame to actually update weights (`new_weight = old_weight - lr × gradient`)

**Overfitting — what it is and how to detect it:** With a tiny dataset, the model risks memorizing specific examples rather than learning generalizable patterns. Detection requires splitting data into training and validation sets and tracking both losses separately. If training loss decreases but validation loss starts increasing, the model is memorizing rather than generalizing. Current mini-GPT code doesn't implement this split properly — a known limitation.

**What the model learned:** After 180 steps, token transition probabilities became meaningfully shaped. The embedding table, W_Q, W_K, W_V matrices, feedforward weights, and all other parameters were all updated jointly — nobody told the model separately which part to improve. The entire forward computation graph was adjusted simultaneously.

---

## Key Architectural Decisions Summary

| Decision | Choice Made | Rationale | Real GPT Comparison |
|---|---|---|---|
| Tokenization | Character-level | Simpler, transparent | Subword BPE, vocab ~50k+ |
| Embedding dim | 64 (learning), 16 (mini GPT) | Readable outputs | 768–12,288 |
| Attention heads | 2 | Minimal multi-perspective | 12–96 |
| Transformer blocks | 2 stacked | Shows layering | 12–96+ |
| FFN expansion | 4x embed_dim | Standard transformer design | Same ratio |
| Optimizer | AdamW, lr=0.001 | Standard modern choice | Same |
| Dataset | Word-level sentences / essays | Semantic learning possible | Internet-scale corpora |

---

## PM-Level Insights Mapped to Technical Mechanisms

**Hallucination:** The model optimizes `P(next_token | previous_tokens)`, not truth. When a prompt falls outside the statistical distribution of training data — a distribution gap — the model still produces fluent continuations because it always generates probability distributions. Fluency does not imply correctness. Grounding through retrieval (RAG) reduces hallucination by anchoring generation to retrieved context, not just learned distributions.

**Context window:** The `(B, T, C)` tensor shape makes this tangible. T is sequence length. Attention complexity scales with T² — every token attends to every other token. Doubling context length quadruples attention computation. This is why long context windows are expensive, why frontier models' long-context capabilities are engineering achievements, and why context window management matters in product design.

**Temperature and randomness:** Dividing logits by temperature before softmax changes how "sharp" the probability distribution is. Low temperature (0.5) pushes probability mass toward the most likely token — deterministic, repetitive, safer. High temperature (1.5) flattens the distribution — more random, more creative, more hallucination-prone. Coding copilots and legal AI products use low temperature; creative writing tools use higher. This is a product decision, not a model decision.

**Cost:** Three drivers — vocabulary size (scores per position), sequence length (tokens processed), and model depth (layers × parameters per layer). Large vocabulary + long context + deep model = expensive inference. Knowing these relationships lets a PM make informed decisions about model selection, context truncation strategies, and caching.

**Latency:** Dominated by matrix multiplications. More layers, larger dimensions, longer sequences → slower. Quantization, distillation, and speculative decoding are engineering approaches to reduce inference latency without full model retraining. As a PM, these tradeoffs between quality and latency are core product decisions.

**Training data shapes behavior:** The distribution of training data determines what the model "knows," what stylistic patterns it imitates, and where it hallucinates. A model trained on medical literature behaves differently than one trained on legal documents even with the same architecture. Fine-tuning, RLHF, and careful data curation are all ways of shifting model behavior — but they're downstream of the fundamental statistical patterns encoded during pretraining.

**Alignment and safety:** A raw pretrained model optimizes for next-token prediction, not helpfulness or safety. RLHF (Reinforcement Learning from Human Feedback) adds a human preference signal — human raters rank outputs, a reward model learns these preferences, and the language model is then fine-tuned to maximize predicted human approval. This is what separates raw GPT from ChatGPT-like assistants. The architecture doesn't change; the training objective changes.

---

## Day 7 — Positional Encoding, Causal Masking, Generation, and Decoding

This is where the model became a real GPT-style system. Four things were added or properly implemented for the first time: positional embeddings, true causal masking inside the transformer, autoregressive text generation, and temperature + top-k decoding. The training setup was also significantly upgraded.

---

### Architectural Upgrades

**Hyperparameters revised:**

```python
embed_dim = 64
block_size = 64    # context window
batch_size = 16
max_iters = 3000
learning_rate = 0.001
```

`block_size` is the context window. During training, each input sequence is exactly 64 tokens long. During generation, the model always feeds only the last 64 tokens — older context gets dropped. This is a hard architectural constraint, not a runtime setting.

**Attention heads upgraded from 2 to 4.** With 4 heads and `embed_dim = 64`, each head operates on a 16-dimensional subspace. The four heads specialize independently through training, not programming.

**4 stacked transformer blocks (up from 2).** Each block refines representations further. Early blocks capture local patterns; later blocks develop more abstract contextual relationships.

**Dropout added (0.1)** after both attention and feedforward within each block, preventing the model from overfitting specific neurons. Switched off during inference via `model.eval()`.

**Final LayerNorm added** before the output projection.

---

### Positional Embeddings

```python
# In MiniGPT.__init__:
self.position_embedding = nn.Embedding(block_size, embed_dim)

# In MiniGPT.forward:
positions = torch.arange(T, device=x.device)
position_embeddings = self.position_embedding(positions)
x = token_embeddings + position_embeddings
```

**Why this was needed:** Attention is permutation-invariant. The core operation `softmax(QK^T/sqrt(d_k))V` treats input as a set of vectors, not a sequence. "Dog bites man" and "Man bites dog" produce nearly identical representations without positional information.

**How it works:** A second learnable lookup table maps each position index (0 through 63) to a 64-dimensional vector. These are added elementwise to token embeddings before any transformer block. The model receives a combined signal: what the word is and where it appears. The same word "cat" at position 0 and position 5 now produces different final representations.

**Decision — learned vs sinusoidal:** The original Attention Is All You Need paper used fixed sine/cosine waves. Modern GPTs use learned positional embeddings — a second `nn.Embedding` that trains alongside everything else. I used learned embeddings. The tradeoff: simpler to implement, but the model cannot generalize beyond `block_size` positions at inference time.

---

### Causal Masking (Properly Implemented)

```python
# Inside MiniTransformerBlock.forward:
T = x.size(1)

mask = torch.triu(
    torch.ones(T, T),
    diagonal=1
).bool().to(x.device)

attn_output, _ = self.attention(x, x, x, attn_mask=mask)
```

**What this does:** `torch.triu(..., diagonal=1)` creates an upper-triangular matrix of ones — the positions above the main diagonal. When passed as `attn_mask` to `nn.MultiheadAttention`, those positions get set to `-inf` before softmax, producing zero attention weight. Token at position 3 can attend to positions 0, 1, 2, 3 but not 4, 5, 6.

**Why the earlier implementation was incomplete:** In tokenizer.py (Day 3), causal masking was implemented manually using `masked_fill` on raw scores. That worked conceptually but was never connected to the full training model in mini_gpt.py. Here it is embedded inside every transformer block, applied dynamically based on actual sequence length, and passed correctly to PyTorch's MultiheadAttention via `attn_mask`. The mask is regenerated every forward pass so it works for any sequence length up to `block_size`.

---

### Training Upgrades: Proper Batching and Validation Split

```python
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

def get_batch(split):
    data_source = train_data if split == "train" else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i : i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1 : i + block_size + 1] for i in ix])
    return x, y
```

**Why random batch sampling matters:** Instead of processing the dataset sequentially, `get_batch` samples 16 random starting positions per step. This prevents the model from memorizing sentence order and forces generalizable pattern learning.

**Why the validation split matters:** Every 200 steps, loss is evaluated on held-out data with `torch.no_grad()`. If training loss decreases while validation loss increases or stalls, the model is memorizing rather than generalizing. This was missing from the Day 6 setup — there was no way to detect overfitting.

**Model checkpoint saved after training:**

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size
}, "mini_gpt_checkpoint.pth")
```

The checkpoint saves weights plus vocabulary mappings so the model can be reloaded for generation without re-training.

---

### Autoregressive Text Generation

```python
def generate(model, start_text, max_new_tokens=60, temperature=0.8, top_k=20):

    model.eval()
    tokens = start_text.split()
    encoded = [stoi[word] for word in tokens if word in stoi]

    x = torch.tensor(encoded).unsqueeze(0)

    for _ in range(max_new_tokens):
        x_cond = x[:, -block_size:]          # context window truncation

        with torch.no_grad():
            logits = model(x_cond)
            logits = logits[:, -1, :]         # last position logits only

            logits = logits / temperature
            values, indices = torch.topk(logits, top_k)
            probs = F.softmax(values, dim=-1)
            sampled_index = torch.multinomial(probs, num_samples=1)
            next_token = indices.gather(-1, sampled_index)

        x = torch.cat([x, next_token], dim=1)

    return " ".join([itos[t] for t in x[0].tolist()])
```

**What autoregressive generation means:** Training predicts all next tokens in parallel. Generation produces one token at a time, appends it, and feeds the extended sequence back in to predict the next. `logits[:, -1, :]` — after a forward pass over the context, only the last position matters. That is the model's distribution over what comes next given everything before.

**Context window truncation `x[:, -block_size:]`:** As generation extends beyond 64 tokens, older context is simply dropped. The model has no memory of anything beyond the window.

The limit is hard for two reasons. First, the position embedding table is `nn.Embedding(64, embed_dim)` — it has exactly 64 rows, indexed 0 to 63. If a sequence of length 65 arrives, `torch.arange(T)` generates index 64, which does not exist in the table — PyTorch throws an index error at runtime. Second, even if you patched the table, the model was trained exclusively on sequences of length 64. Its weights encode attention patterns and feedforward transformations calibrated to that length. Sequences beyond 64 are outside the training distribution; the model has never learned to handle them and would produce garbage. Extending context windows in real LLMs requires architectural changes (rotary position embeddings, sliding window attention, ring attention for very long contexts) plus retraining — not just increasing a number.

---

### Temperature Scaling

```python
logits = logits / temperature
```

Dividing logits by temperature before softmax changes distribution sharpness. Low temperature makes the distribution more peaked — near-deterministic, repetitive, safer. High temperature flattens it — more diverse and surprising, more likely to be wrong. Temperature = 1.0 leaves logits unchanged.

Default used: `temperature = 0.8` — slightly conservative, leaning toward likely tokens while retaining some diversity.

**Product relevance:** Coding copilots and legal AI use low temperature (reliability). Creative writing tools use higher temperature. Customer support bots stay low to avoid inventing answers. This is a product configuration decision, not a model capability decision.

---

### Top-K Filtering

```python
values, indices = torch.topk(logits, top_k)
probs = F.softmax(values, dim=-1)
sampled_index = torch.multinomial(probs, num_samples=1)
next_token = indices.gather(-1, sampled_index)
```

Before applying softmax, keep only the 20 highest-logit tokens. Discard the rest. Apply softmax and sample from the resulting distribution.

**Why not sample from the full vocabulary:** The long tail of low-probability tokens has non-trivial aggregate probability mass. Sampling from it occasionally produces completely nonsensical words. Top-k confines sampling to tokens the model actually considers plausible.

**Why not always take top-1 (greedy):** Greedy decoding selects the single most probable token at every step. This produces deterministic, repetitive, looping outputs — the model gets stuck in high-probability cycles, generating the same phrases over and over. Sampling with top-k introduces controlled randomness that breaks these loops.

| Decoding strategy | Behavior | Risk |
|---|---|---|
| Greedy (top-1) | Deterministic, repetitive | Gets stuck in loops |
| Top-k (k=20, temp=0.8) | Diverse, controlled | Occasional odd word choices |
| High temperature (temp=1.5) | Creative, chaotic | High hallucination rate |

---

### The Hallucination Lab — Mini-GPT vs ChatGPT

The Streamlit app runs both models on the same prompt and displays outputs side by side. Mini-GPT was trained on a small corpus of personal writing. ChatGPT has internet-scale pretraining.

**Prompt:** "begin a sentence with A"

**Mini-GPT output:**
> *my career aspirations and will equip me to create data-informed strategies and an acute pressure to prove myself. She combines strong my prior experience and beyond. [continues in a similar vein — fragments drawn verbatim from the training text, recombined incoherently]*

**ChatGPT output:**
> *A new study has shown that exercise can improve cognitive function.*

**What this comparison demonstrates:**

The Mini-GPT output is not random noise — it has grammatical structure, recognizable English syntax, coherent phrases. It fails for three distinct reasons:

**Distribution gap:** The prompt "begin a sentence with A" is a meta-instruction about sentence structure. That pattern never appeared in the training essays. The model has no learned response to it. Rather than declining, it generates statistically plausible continuations of its training distribution — MBA prose — regardless of what was asked.

**Objective function mismatch:** There is no retrieval, no fact database, no verification. The model generates the most statistically likely next word given context. Accuracy is not in the loss function. This is not a bug — it is the design.

**Training corpus bleed:** Specific phrases from the training text appear verbatim in the output — the model is not generating novel language, it is recombining fragments it memorized. This happens because the training corpus was small enough that the model memorized portions of it rather than learning generalizable patterns. This is a small-model version of what happens in large LLMs when they reproduce training data verbatim — a memorization problem that scales inversely with corpus size.

**PM insight:** This is what hallucination looks like at the mechanism level. The model is not confused or lying in any intentional sense. It is doing exactly what it was trained to do — maximize next-token probability — and the result happens to be wrong relative to what the user wanted. This is why hallucination is hard to fix with prompt engineering alone. The training objective (statistical likelihood) is fundamentally misaligned with the product objective (accuracy). Grounding through retrieval, citation requirements, or constrained generation are architectural solutions to an architectural problem.

---

## What I Know I Don't Know Yet

- Sinusoidal vs learned positional embeddings in depth (used learned; understand the tradeoff at a high level)
- Multi-head attention internals inside `nn.MultiheadAttention` vs the manual W_Q/W_K/W_V implementation
- Nucleus (top-p) sampling — implemented top-k but not top-p
- RLHF — understand conceptually but have not implemented
- CNNs, diffusion models — deferred; not core to LLM product work at my current stage
- How gradient checkpointing and quantization reduce training/inference costs
- Speculative decoding and why it speeds up inference without changing outputs

---

## How to Run

```bash
# Prerequisites
python3 -m venv venv
source venv/bin/activate
pip install torch

# Run each phase independently
python3 tokenizer.py           # Days 1-3: tokenization, embeddings, manual attention
python3 day4_transformer.py    # Day 4: transformer block
python3 mini_gpt.py            # Days 5-6: mini GPT + training
python3 day7_generation.py     # Day 7: full GPT with positional encoding, causal masking, generation

# For the Streamlit comparison app
pip install streamlit openai
streamlit run app.py
```

The Day 7 script will prompt for input interactively after training. Type a word or phrase that exists in the training vocabulary to generate continuations. Type `exit` to quit.

---

## What I Would Tell Another PM Starting This

The hardest conceptual jump is attention. It helps to read it as: "For each token, compute relevance scores against all other tokens using dot products, normalize with softmax so they sum to 1, then use those scores to take a weighted average of all tokens' value vectors." Once that clicks, the rest is engineering scaffolding.

The second hardest is accepting that none of the intelligence in GPT is explicitly programmed. No one encoded grammar rules, reasoning patterns, or factual knowledge. All of it emerged from optimizing next-token prediction at massive scale. This is both the most impressive and most dangerous property of these systems — the model's beliefs are implicit statistical patterns, not verified knowledge.

The most useful thing this project did for me as a PM was make the word hallucination specific. Before building this, hallucination was a vague failure mode. After running the Streamlit comparison and watching my Mini-GPT produce grammatically coherent but completely wrong output in response to a simple prompt, I understand exactly what is happening mechanically: the model optimizes for statistical likelihood, not truth, and when a prompt falls outside its training distribution, it generates fluent text that happens to be wrong. That is not a bug that better prompting fixes. It is the objective function.
