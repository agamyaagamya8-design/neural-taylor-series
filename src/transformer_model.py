"""
transformer_model.py — Transformer Seq2Seq for Neural Taylor Series
====================================================================
Architecture : Encoder–Decoder Transformer (PyTorch nn.Transformer)
Tokenisation : Character-level  (char2idx / idx2char)
Special tokens:
    <PAD>  index 0 — padding
    <s>    index 1 — start-of-sequence (decoder input)
    <e>    index 2 — end-of-sequence   (training target tail)
    <UNK>  index 3 — unknown character

Key Hyperparameters (see CONFIG dict below):
    EMB_DIM     : int  — embedding / model dimension (d_model)
    N_HEADS     : int  — number of attention heads (must divide EMB_DIM)
    N_LAYERS    : int  — number of encoder AND decoder layers
    FFN_DIM     : int  — feed-forward hidden dimension inside Transformer
    DROPOUT     : float— dropout probability
    BATCH_SIZE  : int  — training batch size
    LR          : float— Adam learning rate
    EPOCHS      : int  — training epochs
    TEACHER_FORCE: bool— use teacher forcing during training (recommended)
    MAX_LEN     : int  — hard cap on sequence length (overridden by data)
"""

import os, math, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG = dict(
    DATA_FILE     = "data/dataset.txt",
    CHECKPOINT_DIR= "outputs/",
    EMB_DIM       = 128,      # d_model: embedding dimension
    N_HEADS       = 4,        # attention heads  (EMB_DIM % N_HEADS == 0)
    N_LAYERS      = 3,        # encoder & decoder layers
    FFN_DIM       = 256,      # feed-forward inner dim
    DROPOUT       = 0.1,
    BATCH_SIZE    = 32,
    LR            = 3e-4,     # Adam learning rate
    EPOCHS        = 150,      # increase for better accuracy
    TEACHER_FORCE = True,
    MAX_LEN       = 256,      # hard cap; actual max set from data
    SEED          = 42,
)
# ───────────────────────────────────────────────────────────────────────────────

random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
torch.manual_seed(CONFIG["SEED"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "<e>"
UNK_TOKEN = "<UNK>"
SPECIAL   = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


# ── 1. Vocabulary ──────────────────────────────────────────────────────────────

def build_vocab(pairs):
    """Build char2idx / idx2char from all characters in dataset."""
    chars = set()
    for src, tgt in pairs:
        chars.update(src)
        chars.update(tgt)
    vocab = SPECIAL + sorted(chars)
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char  = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def encode(text, char2idx, add_sos=False, add_eos=False):
    seq = [char2idx.get(c, UNK_IDX) for c in text]
    if add_sos: seq = [SOS_IDX] + seq
    if add_eos: seq = seq + [EOS_IDX]
    return seq


def decode(indices, idx2char):
    chars = []
    for idx in indices:
        if idx == EOS_IDX:
            break
        if idx in (PAD_IDX, SOS_IDX):
            continue
        chars.append(idx2char.get(idx, "?"))
    return "".join(chars)


# ── 2. Dataset ─────────────────────────────────────────────────────────────────

def load_pairs(data_file):
    """Read dataset.txt → list of (src_str, tgt_str)."""
    pairs = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if " -> " in line:
                src, tgt = line.split(" -> ", 1)
                pairs.append((src.strip(), tgt.strip()))
    return pairs


class TaylorDataset(Dataset):
    """
    Each item:
        src      : [src_len]   — encoded source chars
        tgt_in   : [tgt_len]   — decoder input  (SOS + target)
        tgt_out  : [tgt_len]   — decoder target (target + EOS)
    All padded to (max_src_len, max_tgt_len).
    """
    def __init__(self, pairs, char2idx, max_src, max_tgt):
        self.data     = pairs
        self.char2idx = char2idx
        self.max_src  = max_src
        self.max_tgt  = max_tgt + 2   # +2 for SOS / EOS tokens

    def __len__(self): return len(self.data)

    def pad(self, seq, length):
        seq = seq[:length]
        return seq + [PAD_IDX] * (length - len(seq))

    def __getitem__(self, idx):
        src_str, tgt_str = self.data[idx]
        src    = self.pad(encode(src_str, self.char2idx), self.max_src)
        tgt_in = self.pad(encode(tgt_str, self.char2idx, add_sos=True), self.max_tgt)
        tgt_out= self.pad(encode(tgt_str, self.char2idx, add_eos=True), self.max_tgt)
        return (
            torch.tensor(src,     dtype=torch.long),
            torch.tensor(tgt_in,  dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long),
        )


# ── 3. Positional Encoding ─────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x : (batch, seq_len, d_model)
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Extend PE buffer on-the-fly if sequence exceeds pre-built table
            d_model = self.pe.size(2)
            extra = seq_len - self.pe.size(1)
            pos = torch.arange(self.pe.size(1), seq_len,
                               device=x.device).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2, device=x.device).float()
                            * (-math.log(10000.0) / d_model))
            new_pe = torch.zeros(1, extra, d_model, device=x.device)
            new_pe[0, :, 0::2] = torch.sin(pos * div)
            new_pe[0, :, 1::2] = torch.cos(pos * div)
            self.pe = torch.cat([self.pe, new_pe], dim=1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# ── 4. Transformer Seq2Seq Model ───────────────────────────────────────────────

class TransformerSeq2Seq(nn.Module):
    """
    Full encoder-decoder Transformer.

    Args:
        vocab_size  : total number of tokens (chars + special)
        emb_dim     : d_model (embedding + attention dimension)
        n_heads     : number of multi-head attention heads
        n_layers    : number of encoder AND decoder blocks
        ffn_dim     : feed-forward network hidden size inside each block
        dropout     : dropout probability
        max_len     : maximum sequence length for positional encoding
    """
    def __init__(self, vocab_size, emb_dim, n_heads, n_layers, ffn_dim,
                 dropout, max_len):
        super().__init__()
        self.emb_dim = emb_dim

        # Shared embedding table for src and tgt
        self.src_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.tgt_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(emb_dim, max_len, dropout)

        self.transformer = nn.Transformer(
            d_model         = emb_dim,
            nhead           = n_heads,
            num_encoder_layers = n_layers,
            num_decoder_layers = n_layers,
            dim_feedforward = ffn_dim,
            dropout         = dropout,
            batch_first     = True,    # (batch, seq, dim)  — PyTorch ≥ 1.9
        )
        self.fc_out = nn.Linear(emb_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_pad_mask(self, seq):
        """True where token is PAD → those positions are ignored in attention."""
        return seq == PAD_IDX  # (batch, seq_len)

    def make_causal_mask(self, sz):
        """Upper-triangular mask to prevent decoder from attending to future tokens."""
        return torch.triu(torch.ones(sz, sz, device=DEVICE), diagonal=1).bool()

    def encode(self, src):
        src_key_pad = self.make_pad_mask(src)
        src_emb = self.pos_enc(self.src_emb(src) * math.sqrt(self.emb_dim))
        # nn.Transformer.encoder expects (batch, seq, d_model) with batch_first=True
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_pad)
        return memory, src_key_pad

    def decode(self, tgt, memory, src_key_pad):
        tgt_len = tgt.size(1)
        tgt_mask     = self.make_causal_mask(tgt_len)
        tgt_key_pad  = self.make_pad_mask(tgt)
        tgt_emb = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.emb_dim))
        out = self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask            = tgt_mask,
            tgt_key_padding_mask= tgt_key_pad,
            memory_key_padding_mask = src_key_pad,
        )
        return self.fc_out(out)   # (batch, tgt_len, vocab_size)

    def forward(self, src, tgt):
        memory, src_key_pad = self.encode(src)
        return self.decode(tgt, memory, src_key_pad)


# ── 5. Training ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_tokens = 0, 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
        optimizer.zero_grad()
        logits = model(src, tgt_in)              # (B, T, V)
        # Flatten for cross-entropy
        loss = criterion(
            logits.reshape(-1, logits.size(-1)), # (B*T, V)
            tgt_out.reshape(-1),                 # (B*T,)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


# ── 6. Inference — Autoregressive Decoding ────────────────────────────────────

@torch.no_grad()
def predict(model, src_str, char2idx, idx2char, max_tgt_len=200):
    """
    Autoregressively decode the Taylor series for a given expression string.

    Steps:
        1. Encode source string → memory
        2. Start decoder with SOS token
        3. At each step sample the argmax (greedy decoding)
        4. Append predicted token; stop when EOS is generated
    """
    model.eval()
    src = encode(src_str, char2idx)
    src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, src_len)
    memory, src_key_pad = model.encode(src)

    dec_input    = torch.tensor([[SOS_IDX]], dtype=torch.long).to(DEVICE)
    output_ids   = []
    repeat_count = 0
    last_id      = None

    for _ in range(max_tgt_len):
        logits  = model.decode(dec_input, memory, src_key_pad)  # (1, t, V)
        next_id = logits[0, -1].argmax(-1).item()
        if next_id == EOS_IDX:
            break
        # Repetition guard: abort if same token repeats 12+ times consecutively
        if next_id == last_id:
            repeat_count += 1
            if repeat_count >= 12:
                break
        else:
            repeat_count = 0
        last_id = next_id
        output_ids.append(next_id)
        dec_input = torch.cat(
            [dec_input, torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)], dim=1
        )

    return decode(output_ids, idx2char)


# ── 7. Main ────────────────────────────────────────────────────────────────────

def main():
    cfg = CONFIG
    os.makedirs(cfg["CHECKPOINT_DIR"], exist_ok=True)

    # -- Load data
    pairs = load_pairs(cfg["DATA_FILE"])
    if not pairs:
        raise FileNotFoundError(
            f"Dataset not found at '{cfg['DATA_FILE']}'. Run datagen.py first."
        )
    print(f"Loaded {len(pairs)} expression pairs.")

    # -- Build vocabulary
    char2idx, idx2char = build_vocab(pairs)
    vocab_size = len(char2idx)
    print(f"Vocabulary size: {vocab_size} characters")

    # -- Compute max lengths from data (+ safety cap)
    max_src = min(max(len(s) for s, _ in pairs), cfg["MAX_LEN"])
    max_tgt = min(max(len(t) for _, t in pairs), cfg["MAX_LEN"])
    print(f"Max src len: {max_src}  |  Max tgt len: {max_tgt}")

    # -- DataLoader
    dataset    = TaylorDataset(pairs, char2idx, max_src, max_tgt)
    dataloader = DataLoader(dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)

    # -- Model
    model = TransformerSeq2Seq(
        vocab_size = vocab_size,
        emb_dim    = cfg["EMB_DIM"],
        n_heads    = cfg["N_HEADS"],
        n_layers   = cfg["N_LAYERS"],
        ffn_dim    = cfg["FFN_DIM"],
        dropout    = cfg["DROPOUT"],
        max_len    = max(max_src, max_tgt) + 100,  # generous headroom for inference
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # -- Optimizer & loss (ignore PAD in loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                           factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # -- Training loop
    print(f"\nTraining for {cfg['EPOCHS']} epochs …\n")
    for epoch in range(1, cfg["EPOCHS"] + 1):
        loss = train_epoch(model, dataloader, optimizer, criterion)
        scheduler.step(loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{cfg['EPOCHS']}  |  Loss: {loss:.4f}")
            # Quick sanity check on first pair
            src_str, tgt_str = pairs[0]
            pred = predict(model, src_str, char2idx, idx2char, max_tgt_len=max_tgt + 20)
            pred_show = pred[:80] + "…" if len(pred) > 80 else pred
            tgt_show  = tgt_str[:80] + "…" if len(tgt_str) > 80 else tgt_str
            print(f"  Sample  →  Input : {src_str}")
            print(f"           Target : {tgt_show}")
            print(f"           Pred   : {pred_show}\n")

    # -- Save checkpoint
    ckpt_path = os.path.join(cfg["CHECKPOINT_DIR"], "transformer_taylor.pt")
    torch.save({
        "model_state": model.state_dict(),
        "char2idx":    char2idx,
        "idx2char":    idx2char,
        "max_src":     max_src,
        "max_tgt":     max_tgt,
        "config":      cfg,
    }, ckpt_path)
    print(f"Checkpoint saved to '{ckpt_path}'")

    # -- Final evaluation on all pairs
    print("\n── Final Predictions ──────────────────────────────────────────")
    correct = 0
    for src_str, tgt_str in pairs:
        pred = predict(model, src_str, char2idx, idx2char, max_tgt_len=max_tgt + 20)
        match = "✓" if pred.strip() == tgt_str.strip() else "✗"
        if match == "✓": correct += 1
        print(f"{match}  {src_str:25s}  →  {pred}")
    print(f"\nExact match accuracy: {correct}/{len(pairs)} = {correct/len(pairs):.1%}")


if __name__ == "__main__":
    main()



