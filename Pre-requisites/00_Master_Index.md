# 🚀 Gen AI Prerequisites — Master Index

> A complete self-study guide for learning Generative AI from scratch. Five topic files, each with concepts explained, code examples, and visual explanations.

---

## 📚 Files in This Study Pack

| # | File | Topics Covered |
|---|---|---|
| 1 | `01_Python_Programming.md` | Variables, OOP, NumPy, Pandas, Generators |
| 2 | `02_Machine_Learning_Fundamentals.md` | Supervised/Unsupervised ML, Loss functions, Optimization |
| 3 | `03_Deep_Learning_Concepts.md` | Neural Nets, CNNs, RNNs, PyTorch, Backprop |
| 4 | `04_Understanding_Transformers.md` | Self-Attention, BERT, GPT, LoRA, RLHF |
| 5 | `05_NLP_Basics.md` | Tokenization, Embeddings, HuggingFace, RAG |

---

## 🗺️ Recommended Learning Path

```
Week 1-2:  Python Programming        → 01_Python_Programming.md
Week 3-4:  Machine Learning          → 02_Machine_Learning_Fundamentals.md
Week 5-6:  Deep Learning             → 03_Deep_Learning_Concepts.md
Week 7:    NLP Basics                → 05_NLP_Basics.md
Week 8-9:  Transformers              → 04_Understanding_Transformers.md
Week 10+:  Build Gen AI Projects!
```

---

## ⚡ Quick Concept Map

```
PYTHON
  └── NumPy (arrays, math)
  └── Pandas (data)
  └── OOP (classes → PyTorch models)
        ↓
MACHINE LEARNING
  └── Loss functions (cross-entropy → LLM objective)
  └── Gradient descent → Adam optimizer
  └── Overfitting → Dropout, regularization
        ↓
DEEP LEARNING
  └── Neural network layers
  └── Backpropagation
  └── Layer Normalization  ──┐
  └── Residual connections ──┤ Used in every Transformer
  └── PyTorch training loop ─┘
        ↓
NLP BASICS
  └── Tokenization (BPE, WordPiece)
  └── Word embeddings (Word2Vec → BERT)
  └── Text classification, NER, QA
  └── HuggingFace ecosystem
        ↓
TRANSFORMERS
  └── Self-Attention (Q, K, V)
  └── Multi-Head Attention
  └── Encoder (BERT) vs Decoder (GPT)
  └── Fine-tuning (SFT, LoRA, RLHF)
  └── Inference (Temperature, Top-p)
        ↓
GEN AI APPLICATIONS
  └── RAG (Retrieval-Augmented Generation)
  └── Agents & Tool Use
  └── Prompt Engineering
  └── Stable Diffusion / Image Gen
  └── LangChain / LlamaIndex
```

---

## 🔑 Most Critical Concepts (The Core 10)

1. **Python Classes & OOP** — Every PyTorch model is a class
2. **NumPy matrix multiplication** — The core operation in all of ML
3. **Cross-entropy loss** — Training objective for every LLM
4. **Gradient descent & Adam** — How all neural networks learn
5. **Backpropagation** — How gradients flow through deep networks
6. **Layer Normalization + Residual connections** — Inside every Transformer block
7. **Tokenization (BPE)** — How text becomes numbers for LLMs
8. **Self-Attention (Q, K, V)** — The core innovation of Transformers
9. **Softmax + Temperature** — How LLMs generate text
10. **HuggingFace API** — Standard toolkit for all Gen AI work

---

## 🛠️ Essential Libraries to Install

```bash
# Core ML/DL
pip install numpy pandas matplotlib scikit-learn
pip install torch torchvision torchaudio   # PyTorch

# NLP / Gen AI
pip install transformers datasets tokenizers
pip install sentence-transformers
pip install peft accelerate bitsandbytes   # Fine-tuning

# Vector DBs / RAG
pip install faiss-cpu chromadb langchain

# NLP utilities
pip install nltk spacy textblob
python -m spacy download en_core_web_sm
```

---

## 📖 Recommended YouTube Channels

| Channel | Best For |
|---|---|
| Andrej Karpathy | Deep Learning, Transformers from scratch |
| 3Blue1Brown | Math intuition (Linear Algebra, Calculus) |
| Sentdex | Python + ML basics |
| Yannic Kilcher | Paper walkthroughs |
| AI Explained | Gen AI news and concepts |
| HuggingFace | Official tutorials |

---

*Happy Learning! 🎯 Consistency > Intensity — 1 hour/day beats 10 hours/weekend.*
