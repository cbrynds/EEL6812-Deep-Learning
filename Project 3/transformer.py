import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from plotting_utils import plot_attention, plot_avg_attention

def encode(s):
    global stoi
    return [stoi[c] for c in s]

def decode(ids):
    global itos
    return "".join([itos[i] for i in ids])

#  Dataset for next-character prediction
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y
    

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T]
    
# Transformer block with stored attention weights
class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None  # [B, heads, T, T]

    def forward(self, x, attn_mask=None):
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn_weights = attn_weights

        x = self.ln1(x + self.dropout(attn_out))
        x = self.ln2(x + self.dropout(self.ff(x)))
        return x
    
# Transformer language model
class CharTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_ff=256, max_len=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, T, device):
        # shape [T, T], True entries are masked
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(self, x, targets=None):
        # x: [B, T]
        B, T = x.shape
        h = self.token_emb(x)              # [B, T, D]
        h = self.pos_enc(h)

        causal_mask = self.generate_causal_mask(T, x.device)

        for layer in self.layers:
            h = layer(h, attn_mask=causal_mask)

        h = self.ln_f(h)
        logits = self.head(h)              # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T)
            )

        return logits, loss
    
@torch.no_grad()
def evaluate(model, loader, max_batches=50, device="cpu"):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

@torch.no_grad()
def generate(model, start_text, max_new_tokens=200):
    model.eval()
    x = torch.tensor([encode(start_text)], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -block_size:]  # keep last block_size tokens
        logits, _ = model(x_cond)
        next_token_logits = logits[:, -1, :]  # last time step
        probs = F.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    return decode(x[0].tolist())

@torch.no_grad()
def get_attention_maps(model, text_snippet):
    """
    Returns:
      token_ids: [1, T]
      decoded tokens
      attention maps from each layer
    """
    model.eval()

    ids = encode(text_snippet)
    if len(ids) > block_size:
        ids = ids[:block_size]

    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    logits, _ = model(x)

    attn_per_layer = []
    for layer in model.layers:
        # shape: [B, heads, T, T]
        attn_per_layer.append(layer.last_attn_weights.detach().cpu())

    return x.cpu(), attn_per_layer

def main():
    global stoi, itos
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    torch.manual_seed(42)

    #  Download / load dataset
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("Dataset length:", len(text))
    print(text[:500])

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor(encode(text), dtype=torch.long)
    print("Vocabulary size:", vocab_size)
    print(stoi)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    block_size = 64
    batch_size = 64

    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)

    print("Train dataset length:", len(train_dataset))
    print("Val dataset length:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    data_item = next(iter(train_loader))
    print(data_item[1].shape)

    base_num_layers = 2
    ablation_num_layers = [1,2,4]
    ablation_context_length = [32, 64, 128]
    runs = []

    model = CharTransformerLM(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        max_len=block_size,
        dropout=0.1
    ).to(device)

    print(model)

    # =========================================================
    # 9. Training and evaluation
    # =========================================================

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % 200 == 0:
                avg_train = running_loss / 200
                val_loss = evaluate(model, val_loader, max_batches=20, device=device)
                print(f"Epoch {epoch+1}, Step {step+1}, Train Loss: {avg_train:.4f}, Val Loss: {val_loss:.4f}")
                running_loss = 0.0

    # =========================================================
    # 10. Text generation
    # =========================================================

    sample = generate(model, "ROMEO:\n", max_new_tokens=300)
    print(sample)

    # =========================================================
    # 11. Analyze attention maps
    # =========================================================

    snippet = "To be, or not to be, that is the question:"
    x_tokens, attn_maps = get_attention_maps(model, snippet)

    tokens = [itos[i] for i in x_tokens[0].tolist()]
    print("Tokens:", tokens)

    # =========================================================
    # 12. Visualize one attention head
    # =========================================================

    plot_attention(attn_maps, tokens, layer_idx=0, head_idx=0)
    plot_attention(attn_maps, tokens, layer_idx=1, head_idx=1)

    # =========================================================
    # 13. Average attention across heads in a layer
    # =========================================================

    plot_avg_attention(attn_maps, tokens, layer_idx=0)
    plot_avg_attention(attn_maps, tokens, layer_idx=1)

if __name__ == "__main__":
    main()