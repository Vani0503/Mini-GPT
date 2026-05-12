import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------
# Hyperparameters
# -----------------------------------

embed_dim = 64
block_size = 64
batch_size = 16
max_iters = 3000
learning_rate = 0.001

# -----------------------------------
# Transformer Block
# -----------------------------------

class MiniTransformerBlock(nn.Module):

    def __init__(self, embed_dim):

        super().__init__()

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)

        # Small dropout
        self.dropout1 = nn.Dropout(0.1)

        # Feedforward network
        self.ff = nn.Sequential(

            nn.Linear(embed_dim, embed_dim * 4),

            nn.ReLU(),

            nn.Linear(embed_dim * 4, embed_dim)

        )

        # LayerNorm
        self.norm2 = nn.LayerNorm(embed_dim)

        # Small dropout
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):

        T = x.size(1)

        # -----------------------------------
        # Causal Mask
        # Prevent future-token visibility
        # -----------------------------------

        mask = torch.triu(
            torch.ones(T, T),
            diagonal=1
        ).bool().to(x.device)

        # -----------------------------------
        # Self Attention
        # -----------------------------------

        attn_output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=mask
        )

        # -----------------------------------
        # Residual + LayerNorm
        # -----------------------------------

        x = self.norm1(
            x + self.dropout1(attn_output)
        )

        # -----------------------------------
        # Feedforward Network
        # -----------------------------------

        ff_output = self.ff(x)

        # -----------------------------------
        # Residual + LayerNorm
        # -----------------------------------

        x = self.norm2(
            x + self.dropout2(ff_output)
        )

        return x

# -----------------------------------
# Mini GPT
# -----------------------------------

class MiniGPT(nn.Module):

    def __init__(self, vocab_size, embed_dim):

        super().__init__()

        # Token embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            block_size,
            embed_dim
        )

        # Transformer blocks
        self.blocks = nn.Sequential(

            MiniTransformerBlock(embed_dim),
            MiniTransformerBlock(embed_dim),
            MiniTransformerBlock(embed_dim),
            MiniTransformerBlock(embed_dim)

        )

        # Final normalization
        self.final_norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.linear = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(self, x):

        B, T = x.shape

        # -----------------------------------
        # Token Embeddings
        # -----------------------------------

        token_embeddings = self.embedding(x)

        # -----------------------------------
        # Position Embeddings
        # -----------------------------------

        positions = torch.arange(
            T,
            device=x.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # -----------------------------------
        # Combine Embeddings
        # -----------------------------------

        x = token_embeddings + position_embeddings

        # -----------------------------------
        # Transformer Blocks
        # -----------------------------------

        x = self.blocks(x)

        # -----------------------------------
        # Final Normalization
        # -----------------------------------

        x = self.final_norm(x)

        # -----------------------------------
        # Final Logits
        # -----------------------------------

        logits = self.linear(x)

        return logits

# -----------------------------------
# Load Dataset
# -----------------------------------

print("Day 7 - Personalized Mini GPT")

with open(
    "training_corpus.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

# -----------------------------------
# Tokenization
# -----------------------------------

words = text.split()

vocab = sorted(list(set(words)))

stoi = {
    word: i
    for i, word in enumerate(vocab)
}

itos = {
    i: word
    for word, i in stoi.items()
}

vocab_size = len(vocab)

print("\nVocabulary Size:")
print(vocab_size)

# -----------------------------------
# Encode Text
# -----------------------------------

encoded = [
    stoi[word]
    for word in words
]

data = torch.tensor(encoded)

print("\nTotal Tokens:")
print(len(data))

# -----------------------------------
# Train / Validation Split
# -----------------------------------

split_idx = int(0.8 * len(data))

train_data = data[:split_idx]

val_data = data[split_idx:]

print("\nTrain Tokens:")
print(len(train_data))

print("\nValidation Tokens:")
print(len(val_data))

# -----------------------------------
# Batch Sampling
# -----------------------------------

def get_batch(split):

    data_source = (
        train_data
        if split == "train"
        else val_data
    )

    # Random starting positions
    ix = torch.randint(
        len(data_source) - block_size,
        (batch_size,)
    )

    # Input batch
    x = torch.stack([

        data_source[i : i + block_size]

        for i in ix
    ])

    # Target batch
    y = torch.stack([

        data_source[i + 1 : i + block_size + 1]

        for i in ix
    ])

    return x, y

# -----------------------------------
# Create Model
# -----------------------------------

model = MiniGPT(
    vocab_size,
    embed_dim
)

print("\nModel created!")

# -----------------------------------
# Loss Function
# -----------------------------------

loss_fn = nn.CrossEntropyLoss()

# -----------------------------------
# Optimizer
# -----------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print("\nStarting training...\n")

# -----------------------------------
# Training Loop
# -----------------------------------

for step in range(max_iters):

    # -----------------------------------
    # Get Training Batch
    # -----------------------------------

    x, y = get_batch("train")

    # -----------------------------------
    # Forward Pass
    # -----------------------------------

    logits = model(x)

    # -----------------------------------
    # Reshape Logits
    # -----------------------------------

    logits = logits.view(
        -1,
        vocab_size
    )

    # -----------------------------------
    # Flatten Targets
    # -----------------------------------

    targets = y.view(-1)

    # -----------------------------------
    # Compute Loss
    # -----------------------------------

    loss = loss_fn(
        logits,
        targets
    )

    # -----------------------------------
    # Backpropagation
    # -----------------------------------

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # -----------------------------------
    # Validation Tracking
    # -----------------------------------

    if step % 200 == 0:

        with torch.no_grad():

            val_x, val_y = get_batch("val")

            val_logits = model(val_x)

            val_logits = val_logits.view(
                -1,
                vocab_size
            )

            val_targets = val_y.view(-1)

            val_loss = loss_fn(
                val_logits,
                val_targets
            )

        print(

            f"Step {step} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f}"

        )

print("\nTraining complete!")

# -----------------------------------
# Save Model Checkpoint
# -----------------------------------

torch.save(

    {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size
    },

    "mini_gpt_checkpoint.pth"

)

print("\nModel checkpoint saved!")

# -----------------------------------
# Text Generation Function
# -----------------------------------

def generate(
    model,
    start_text,
    max_new_tokens=60,
    temperature=0.8,
    top_k=20
):

    model.eval()

    # -----------------------------------
    # Encode Prompt
    # -----------------------------------

    tokens = start_text.split()

    encoded = [

        stoi[word]

        for word in tokens

        if word in stoi
    ]

    # Handle unknown prompts
    if len(encoded) == 0:

        return "Prompt words not found in vocabulary."

    x = torch.tensor(encoded).unsqueeze(0)

    # -----------------------------------
    # Generate Tokens
    # -----------------------------------

    for _ in range(max_new_tokens):

        # Keep recent context only
        x_cond = x[:, -block_size:]

        with torch.no_grad():

            logits = model(x_cond)

            # Last token logits
            logits = logits[:, -1, :]

            # -----------------------------------
            # Temperature Scaling
            # -----------------------------------

            logits = logits / temperature

            # -----------------------------------
            # Top-K Filtering
            # -----------------------------------

            values, indices = torch.topk(
                logits,
                top_k
            )

            # -----------------------------------
            # Softmax Probabilities
            # -----------------------------------

            probs = F.softmax(
                values,
                dim=-1
            )

            # -----------------------------------
            # Random Sampling
            # -----------------------------------

            sampled_index = torch.multinomial(
                probs,
                num_samples=1
            )

            next_token = indices.gather(
                -1,
                sampled_index
            )

        # Append generated token
        x = torch.cat(
            [x, next_token],
            dim=1
        )

    # -----------------------------------
    # Decode Tokens
    # -----------------------------------

    generated_tokens = x[0].tolist()

    generated_words = [

        itos[token]

        for token in generated_tokens
    ]

    return " ".join(generated_words)

# -----------------------------------
# Interactive Mini GPT
# -----------------------------------

print("\n-----------------------------------")
print("Mini GPT")
print("-----------------------------------")
print("Type 'exit' to quit.\n")

while True:

    prompt = input("Prompt: ")

    if prompt.lower() == "exit":

        print("\nGoodbye!\n")

        break

    generated_output = generate(
        model,
        start_text=prompt,
        max_new_tokens=60,
        temperature=0.8,
        top_k=20
    )

    print("\nMiniGPT Response:\n")

    print(generated_output)

    print("\n-----------------------------------\n")
