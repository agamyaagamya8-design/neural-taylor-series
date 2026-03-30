# 🧠 Neural Taylor Series Generator

A sequence-to-sequence neural network that learns to predict **Taylor series expansions** of mathematical expressions from symbolic input strings — purely from data, without hard-coded rules.

---

## 📁 Project Structure

```
neural-taylor-series/
├─ src/
│   ├─ datagen.py            ← Generate dataset via SymPy
│   ├─ transformer_model.py  ← Transformer Seq2Seq (recommended)
│   └─ lstm_model.py         ← LSTM Seq2Seq (faster on small data)
├─ data/
│   └─ dataset.txt           ← Auto-generated input/output pairs
├─ outputs/                  ← Saved model checkpoints
├─ requirements.txt
└─ README.md
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python src/datagen.py

# 3. Train Transformer (recommended)
python src/transformer_model.py

# 4. OR train LSTM (faster convergence on small datasets)
python src/lstm_model.py
```

---

## 📊 Dataset Format (`data/dataset.txt`)

Each line is an `expression -> Taylor expansion` pair:

```
sin(x) -> x - x**3/6 + x**5/120
cos(x) -> 1 - x**2/2 + x**4/24
exp(x) -> 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120
x**2 -> x**2
```

- Expansions are truncated at **x⁶** (6 terms, configurable via `TAYLOR_ORDER`)
- 24 expressions by default; easily extendable in `datagen.py`

---

## 🏗️ Architecture

### Transformer (`transformer_model.py`)
| Component | Detail |
|---|---|
| Tokenisation | Character-level |
| Encoder | `nn.Transformer` encoder stack |
| Decoder | `nn.Transformer` decoder stack (causal mask) |
| Positional encoding | Sinusoidal (Vaswani et al. 2017) |
| Decoding | Greedy autoregressive |

**Hyperparameters:**

| Param | Default | Description |
|---|---|---|
| `EMB_DIM` | 128 | d_model — embedding & attention dimension |
| `N_HEADS` | 4 | Multi-head attention heads (must divide EMB_DIM) |
| `N_LAYERS` | 3 | Encoder + Decoder layer count |
| `FFN_DIM` | 256 | Feed-forward hidden size inside each block |
| `DROPOUT` | 0.1 | Dropout probability |
| `BATCH_SIZE` | 32 | Training batch size |
| `LR` | 3e-4 | Adam learning rate |
| `EPOCHS` | 150 | Training epochs (increase for better accuracy) |

### LSTM (`lstm_model.py`)
| Component | Detail |
|---|---|
| Encoder | Bidirectional LSTM (fwd+bwd hidden projected) |
| Attention | Bahdanau (additive) attention |
| Decoder | Unidirectional LSTM + attention context |

**Hyperparameters:**

| Param | Default | Description |
|---|---|---|
| `EMB_DIM` | 64 | Character embedding size |
| `HID_DIM` | 256 | LSTM hidden state size |
| `N_LAYERS` | 2 | Stacked LSTM layers |
| `DROPOUT` | 0.3 | Dropout between LSTM layers |
| `LR` | 1e-3 | Adam learning rate |
| `EPOCHS` | 100 | Training epochs |
| `CLIP` | 1.0 | Gradient clipping norm |

---

## 🔑 Key Design Decisions

### Tokenisation
Character-level tokenisation is used — every character (`s`, `i`, `n`, `(`, `x`, `)`, …) is a separate token. This keeps the vocabulary tiny (~40 chars) at the cost of longer sequences.

### Special Tokens
| Token | Index | Role |
|---|---|---|
| `<PAD>` | 0 | Padding to equal-length batches |
| `<s>` | 1 | Start-of-sequence (decoder input) |
| `<e>` | 2 | End-of-sequence (stop signal) |
| `<UNK>` | 3 | Unknown character (fallback) |

### Autoregressive Inference
At inference time the decoder generates **one character at a time**:
1. Feed `<s>` as the first decoder token
2. Take the `argmax` of output logits → next character
3. Append to decoder input; repeat
4. Stop when `<e>` is generated or max length is reached

### Loss
`CrossEntropyLoss` with `ignore_index=PAD_IDX` — padding positions do not contribute to the loss or gradients.

---

## 🔮 Extending the Project

- **More functions**: add entries to `get_functions()` in `datagen.py`
- **Higher order**: increase `TAYLOR_ORDER` in `datagen.py`
- **Beam search**: replace greedy decoding with beam search for better accuracy
- **Larger model**: increase `EMB_DIM`, `N_LAYERS`, `FFN_DIM` for more expressive power
- **Data augmentation**: generate scaled/shifted variants `sin(2x)`, `exp(x/3)`, etc.

---

## 🗂️ Version Control

```bash
git init
git add .
git commit -m "Initial commit: neural Taylor series generator"
git remote add origin <your-repo-URL>
git branch -M main
git push -u origin main
```

---

## 📦 Requirements

```
torch>=2.0.0
sympy>=1.12
numpy>=1.24
```

GPU is used automatically if available (`cuda`), otherwise falls back to CPU.

