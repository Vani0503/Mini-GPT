import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from openai import OpenAI

embed_dim = 64
block_size = 64

class MiniTransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[MiniTransformerBlock(embed_dim) for _ in range(4)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x) + self.position_embedding(torch.arange(T, device=x.device))
        return self.linear(self.final_norm(self.blocks(x)))

@st.cache_resource
def load_model():
    checkpoint = torch.load("mini_gpt_checkpoint.pth", map_location="cpu")
    model = MiniGPT(checkpoint["vocab_size"], embed_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["stoi"], checkpoint["itos"]

def generate(model, stoi, itos, prompt):
    tokens = [stoi[w] for w in prompt.split() if w in stoi]
    if not tokens:
        return "None of those words are in my vocabulary."
    x = torch.tensor(tokens).unsqueeze(0)
    for _ in range(60):
        x_cond = x[:, -block_size:]
        with torch.no_grad():
            logits = model(x_cond)[:, -1, :] / 0.8
            values, indices = torch.topk(logits, 20)
            probs = F.softmax(values, dim=-1)
            next_token = indices.gather(-1, torch.multinomial(probs, 1))
        x = torch.cat([x, next_token], dim=1)
    return " ".join([itos[t] for t in x[0].tolist()[len(tokens):]])

def get_chatgpt(prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

# UI
st.title("Mini-GPT vs ChatGPT")
st.caption("A GPT trained from scratch vs ChatGPT — side by side")

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.text_input("Enter your prompt")

if st.button("Compare") and prompt:
    model, stoi, itos = load_model()
    mini_response = generate(model, stoi, itos, prompt)
    gpt_response = get_chatgpt(prompt)
    st.session_state.history.append({
        "prompt": prompt,
        "mini": mini_response,
        "gpt": gpt_response
    })

# Display full history
for entry in st.session_state.history:
    st.markdown(f"**You:** {entry['prompt']}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mini-GPT")
        st.write(entry["mini"])
    with col2:
        st.subheader("ChatGPT")
        st.write(entry["gpt"])
    st.divider()
