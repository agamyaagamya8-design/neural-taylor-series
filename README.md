# 🧠 Neural Taylor Series Generator

> A Transformer-based Seq2Seq model that **learns to approximate symbolic mathematics** by generating Taylor series expansions — entirely from data, without explicit mathematical rules.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![SymPy](https://img.shields.io/badge/SymPy-1.12+-green)
![Status](https://img.shields.io/badge/Status-Research--Prototype-brightgreen)

---

## 🎯 What Does This Do?

This model takes a mathematical expression and generates its Taylor series expansion:

```
Input  :  sin(x)
Output :  x - x**3/6 + x**5/120

Input  :  exp(x)
Output :  1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120

Input  :  log(1+x)
Output :  x - x**2/2 + x**3/3 - x**4/4 + x**5/5
```

Unlike symbolic engines, this model:

* does **not derive formulas**
* does **not use calculus rules**
* learns purely from **pattern recognition in data**

---

## 💡 Why This Matters

Taylor series are fundamental to:

* numerical computing
* scientific simulations
* embedded systems
* ML approximations

This project explores a deeper question:

> Can neural networks *learn mathematical structure* instead of being programmed with it?

This sits at the intersection of:

* Deep Learning
* Symbolic Mathematics
* Program Synthesis

---

## 🚀 What Makes This Different

Most similar projects:

* Train on fixed datasets
* Use only expansion at `x = 0`
* Learn shallow mappings

### This project introduces:

✅ **Dynamic dataset generation (SymPy-powered)**

* Expressions generated programmatically
* Not static / hand-written

✅ **Multi-function coverage**

* Trigonometric, exponential, logarithmic, rational, polynomial

✅ **Configurable expansion order**

* Easily adjustable Taylor depth

✅ **Full pipeline control**

* Data → Model → Training → Inference

---

## 🏗️ Architecture Overview

```
Expression → Tokenizer → Transformer Encoder → Transformer Decoder → Taylor Series
   "sin(x)"                                       ↓
                                            "x - x**3/6 + ..."
```

### Model Design

| Component           | Details                            |
| ------------------- | ---------------------------------- |
| Tokenization        | Character-level                    |
| Vocabulary          | 31 tokens + special symbols        |
| Encoder             | Multi-head self-attention          |
| Decoder             | Masked attention + cross-attention |
| Positional Encoding | Sinusoidal                         |
| Decoding            | Greedy autoregressive              |

---

## 🧪 Training Insights

| Epoch | Loss  | Interpretation              |
| ----- | ----- | --------------------------- |
| 1     | 2.13  | Random output               |
| 10    | 0.40  | Structure emerging          |
| 30    | 0.09  | Correct patterns forming    |
| 60    | 0.04  | High accuracy               |
| 150   | ~0.01 | Near-perfect reconstruction |

📉 **Fast convergence** due to structured nature of mathematical data.

---

## 📂 Project Structure

```
neural-taylor-series/
├── src/
│   ├── datagen.py            # Dynamic dataset generation (SymPy)
│   ├── transformer_model.py  # Transformer Seq2Seq
│   └── lstm_model.py         # Baseline comparison model
├── data/
│   └── dataset.txt
├── outputs/
│   └── transformer_taylor.pt
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Setup

```bash
git clone https://github.com/yourusername/neural-taylor-series.git
cd neural-taylor-series
pip install -r requirements.txt
```

### 2. Generate Data

```bash
python src/datagen.py
```

### 3. Train

```bash
python src/transformer_model.py
```

---

## 📦 Dataset Details

* ~1000 generated samples
* Format:

  ```
  expression -> Taylor expansion
  ```
* Expansion order: up to **x⁶**
* Generated using **SymPy**

---

## ⚙️ Key Hyperparameters

| Parameter     | Value |
| ------------- | ----- |
| Embedding Dim | 128   |
| Heads         | 4     |
| Layers        | 3     |
| FFN Dim       | 256   |
| Batch Size    | 32    |
| LR            | 3e-4  |
| Epochs        | 150   |

---

## 🔍 How It Actually Learns

Instead of memorizing formulas, the model learns:

* power series patterns
* coefficient relationships
* sign alternations
* polynomial growth structures

This is **pattern abstraction**, not symbolic reasoning.

---

## 🔮 Future Improvements

* Beam search decoding
* Multi-variable Taylor expansions
* Error analysis vs true series
* Attention visualization
* Web interface (input → output live)

---

## 🧠 Key Takeaways

* Neural networks can approximate structured math mappings
* Transformers handle symbolic sequences effectively
* Data quality matters more than model size

---

## 👨‍💻 Author

**Agamya**
First deep learning project — built from scratch 🚀

---

## 📄 License

MIT License

---

## ⭐ Support

If you found this interesting, drop a ⭐ — it genuinely helps visibility.


