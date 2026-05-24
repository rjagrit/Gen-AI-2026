# ⚡ Understanding Transformers (like GPT) — Complete Guide for Gen AI

> **Why Transformers?** The Transformer architecture is the foundation of EVERY modern Gen AI model — GPT, Claude, LLaMA, BERT, T5, Stable Diffusion's text encoder. Understanding it deeply is the single most important concept in Gen AI.

---

## 1. Why Transformers? (The Problem with RNNs)

Before Transformers, NLP used RNNs and LSTMs. They had critical limitations:

| Problem | RNN Issue | Transformer Solution |
|---|---|---|
| Long-range dependencies | Gradients vanish over 100+ steps | Self-attention spans full sequence |
| Sequential processing | Can't parallelize — slow to train | All positions processed in parallel |
| Memory of earlier tokens | Hidden state bottle-neck | Direct access to any past token |

**The paper:** "Attention is All You Need" — Vaswani et al., 2017 (Google Brain)

---

## 2. Tokenization — Text to Numbers

Transformers don't read text — they read **token IDs** (integers).

### What is a Token?
```
"Hello, world!" → ["Hello", ",", " world", "!"]   → [15496, 11, 995, 0]
```

Tokens can be words, subwords, or characters depending on the tokenizer.

### Byte-Pair Encoding (BPE) — Used by GPT
```
Start with characters: ["h", "e", "l", "l", "o"]
Merge most frequent pairs iteratively:
  "h"+"e" → "he"
  "he"+"l" → "hel"
  "hel"+"lo" → "hello"
Final vocab has subword units
```

This handles rare words gracefully — "unhappiness" might be ["un", "happi", "ness"].

### WordPiece — Used by BERT
```
Similar to BPE but uses likelihood instead of frequency
"playing" → ["play", "##ing"]   (## = continuation of word)
```

### Special Tokens
| Token | Purpose |
|---|---|
| `[CLS]` | Classification token — BERT uses output here for classification |
| `[SEP]` | Separator between sentences |
| `[PAD]` | Padding to make sequences same length in a batch |
| `[MASK]` | Masked token for BERT's masked language modeling |
| `<|endoftext|>` | End-of-sequence token for GPT |

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, Generative AI!"
tokens = tokenizer(text)
# {'input_ids': [15496, 11, 2448, ..], 'attention_mask': [1, 1, 1, ..]}

decoded = tokenizer.decode(tokens['input_ids'])
# "Hello, Generative AI!"
```

---

## 3. Embeddings — Numbers to Meaning

### Token Embeddings
Each token ID is mapped to a dense vector. This is a **learned lookup table**.

```
Token ID 15496 ("Hello") → [0.12, -0.45, 0.78, ...]  ← 768-dimensional vector
```

```python
import torch.nn as nn

vocab_size = 50257       # GPT-2 vocabulary size
embed_dim  = 768         # embedding dimension

embedding = nn.Embedding(vocab_size, embed_dim)
# embedding(token_ids) → (batch, seq_len, 768)
```

### Positional Encoding — Injecting Order

Transformers process all tokens in parallel — they have **no inherent sense of order**. We must add position information explicitly.

#### Sinusoidal Positional Encoding (Original Transformer)
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

pos = position in sequence
i   = dimension index
```

```python
import torch
import math

def sinusoidal_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div)   # even dims
    pe[:, 1::2] = torch.cos(position * div)   # odd dims
    return pe
```

#### Learned Positional Embeddings (GPT-2)
```python
# Simply an additional embedding table for positions
pos_embed = nn.Embedding(max_seq_len, embed_dim)
# Train like any other parameter
```

#### RoPE — Rotary Position Embedding (LLaMA, GPT-NeoX)
- Encodes relative position directly into Q/K before attention
- Better length generalization (can handle sequences longer than training)

### Combined Embedding
```
Final Input = Token Embedding + Positional Embedding
Shape: (batch_size, sequence_length, d_model)
```

---

## 4. Self-Attention — The Core Mechanism

Self-attention allows **every token to look at every other token** and decide how much to focus on it.

### Query, Key, Value (Q, K, V)

Think of it like a search engine:
```
Query (Q) = "what am I looking for?"
Key   (K) = "what does each token advertise?"
Value (V) = "what information does each token carry?"

Attention = match Q against all K → use scores to weight V
```

### Attention Computation Step by Step

```
Input: X of shape (seq_len, d_model)

Step 1: Project to Q, K, V
  Q = X × Wq    shape: (seq_len, d_k)
  K = X × Wk    shape: (seq_len, d_k)
  V = X × Wv    shape: (seq_len, d_v)

Step 2: Compute raw attention scores
  scores = Q × Kᵀ     shape: (seq_len, seq_len)
  Each score = how much token i attends to token j

Step 3: Scale (prevent vanishing softmax gradients)
  scores = scores / √d_k

Step 4: Apply softmax → attention weights
  weights = softmax(scores)   → each row sums to 1

Step 5: Weighted sum of values
  output = weights × V        shape: (seq_len, d_v)
```

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights
```

### Visualizing Attention

```
Sentence: "The cat sat on the mat because it was tired"

When processing "it", the model attends to:
"cat" → 0.72  ← high attention (it refers to the cat)
"mat" → 0.12
"sat" → 0.08
...
```

---

## 5. Multi-Head Attention

Instead of one attention, run **H parallel attention heads** — each learns different relationships.

```
Input X → Split into H heads → Each head: attention(QᵢKᵢVᵢ)
                             ↓
                    Concatenate all heads
                             ↓
                    Linear projection → output
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        # (B, num_heads, T, d_k)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        out, _ = attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)
        return self.W_o(out)
```

**Why multiple heads?**
- Head 1 might learn syntactic relationships (subject-verb)
- Head 2 might learn coreference ("it" → "cat")
- Head 3 might learn positional relationships (nearby words)

---

## 6. The Transformer Block

Every Transformer model is built from stacked identical blocks.

```
Input
  │
  ▼
┌─────────────────────────────────┐
│   Multi-Head Self-Attention     │
└─────────────────────────────────┘
  │          │
  │    (residual)
  ▼          │
┌─────────────────────────────────┐
│       Add & LayerNorm           │  x = LayerNorm(x + Attention(x))
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│  Feed-Forward Network (FFN)     │  Linear → GELU → Linear
└─────────────────────────────────┘
  │          │
  │    (residual)
  ▼          │
┌─────────────────────────────────┐
│       Add & LayerNorm           │  x = LayerNorm(x + FFN(x))
└─────────────────────────────────┘
  │
  ▼
Output (passed to next block)
```

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual
        x = x + self.drop(self.attn(self.norm1(x), mask))
        # FFN + residual
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
```

---

## 7. Encoder vs Decoder Architectures

### Encoder-Only (e.g., BERT)
```
Input → Tokenize → Embed → [Block×N] → Contextual Representations

Uses BIDIRECTIONAL attention — each token sees all other tokens
Best for: Text classification, NER, extractive QA, embeddings
```

### Decoder-Only (e.g., GPT, Claude, LLaMA)
```
Input → Tokenize → Embed → [Block×N] → Next Token Prediction

Uses CAUSAL (unidirectional) attention — each token sees only LEFT context
Autoregressive: generates one token at a time
Best for: Text generation, chat, code generation
```

### Encoder-Decoder (e.g., T5, BART)
```
Encoder processes input → Decoder cross-attends to encoder output

Encoder: Bidirectional attention
Decoder: Causal self-attention + Cross-attention over encoder
Best for: Translation, summarization, question answering
```

---

## 8. Causal (Masked) Attention

GPT-style models can only attend to **past tokens** during training.

```
Attention mask for sequence "the cat sat":
        the  cat  sat
the  [ 1    0    0  ]   → "the" can only see itself
cat  [ 1    1    0  ]   → "cat" sees "the" and itself
sat  [ 1    1    1  ]   → "sat" sees all

-inf replaces 0 before softmax → effectively zero after softmax
```

```python
def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len))
    # Lower triangular matrix of ones
```

---

## 9. Key Models to Know

### BERT — Bidirectional Encoder (2018, Google)
```
Architecture: Encoder-only, 12 layers, 768 dim, 12 heads (BERT-base)
Pretraining:
  - Masked Language Modeling: predict [MASK] tokens
  - Next Sentence Prediction: is sentence B a continuation of A?
Use cases: Text classification, NER, question answering
```

### GPT-2 / GPT-3 / GPT-4 — Autoregressive Decoder (OpenAI)
```
Architecture: Decoder-only
Pretraining: Next token prediction (language modeling)
GPT-2: 1.5B params, context 1024
GPT-3: 175B params, context 2048
GPT-4: ~1.7T params (estimated), multimodal
```

### T5 — Text-to-Text Transfer Transformer (2020, Google)
```
Architecture: Encoder-Decoder
Unified framework: EVERY task is text-in → text-out
  "translate English to French: Hello" → "Bonjour"
  "summarize: [article]" → "summary"
```

### LLaMA 2 / LLaMA 3 — Meta's Open Source GPT-style
```
Architecture: Decoder-only (like GPT)
Uses: RoPE positional embeddings, SwiGLU activation, GQA
Open weights — can run locally
```

---

## 10. Pretraining, Fine-tuning & Alignment

### Stage 1: Pretraining
```
Data: Massive web text (trillions of tokens)
Objective: Predict next token (GPT) or masked tokens (BERT)
Result: General-purpose language understanding

LLaMA 3: Trained on 15 trillion tokens
```

### Stage 2: Supervised Fine-Tuning (SFT)
```
Data: High-quality instruction-response pairs (thousands to millions)
Objective: Follow instructions, be helpful
Result: Model learns to chat and follow directions
```

### Stage 3: RLHF — Reinforcement Learning from Human Feedback
```
1. Collect human preferences (A vs B response comparisons)
2. Train Reward Model to predict human preference
3. Fine-tune LLM using PPO (policy optimization) to maximize reward
4. Result: More helpful, harmless, and honest responses
Used by: ChatGPT, Claude
```

### Parameter-Efficient Fine-Tuning (PEFT)

Fine-tuning all parameters is expensive. PEFT trains only a small fraction.

#### LoRA — Low-Rank Adaptation
```
Original weight matrix W (d×k)
LoRA: W' = W + BA   where B is (d×r), A is (r×k), r << d

Only A and B are trained (r is typically 4-64)
Reduces trainable params by 99%
```

```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 7,000,000,000 || 0.06%
```

---

## 11. Inference & Generation

### Temperature
```python
# temperature = 0 → greedy (always pick most likely token)
# temperature = 1 → standard sampling
# temperature > 1 → more random/creative

probs = F.softmax(logits / temperature, dim=-1)
```

### Sampling Strategies
```python
# Greedy: always pick argmax
token = logits.argmax(-1)

# Top-k: sample from top k tokens only
top_k_probs, top_k_ids = torch.topk(probs, k=50)
token = top_k_ids[torch.multinomial(top_k_probs, 1)]

# Top-p (Nucleus): sample from tokens summing to probability p
sorted_probs, sorted_ids = torch.sort(probs, descending=True)
cumsum = torch.cumsum(sorted_probs, dim=-1)
# Remove tokens beyond cumulative probability p
```

### Key-Value (KV) Cache
```
During generation, we recompute K and V for all past tokens every step.
KV Cache: store past K and V matrices → only compute new token's K and V
Reduces generation cost from O(n²) to O(n)
```

---

## 12. Transformer Scale — GPT vs GPT-3 vs LLaMA

| Model | Layers | Heads | d_model | Params |
|---|---|---|---|---|
| GPT-2 (small) | 12 | 12 | 768 | 117M |
| GPT-2 (large) | 36 | 20 | 1280 | 774M |
| GPT-3 | 96 | 96 | 12288 | 175B |
| LLaMA 2 7B | 32 | 32 | 4096 | 7B |
| LLaMA 3 70B | 80 | 64 | 8192 | 70B |

---

## ✅ Transformer Checklist for Gen AI

| Concept | Why It Matters |
|---|---|
| Tokenization (BPE) | First step in any LLM pipeline |
| Token + Positional Embeddings | How text becomes numbers |
| Self-Attention (Q, K, V) | Core computation in every LLM |
| Multi-Head Attention | Multiple relationship patterns |
| Transformer Block (Attn + FFN + LN) | The repeating unit of all LLMs |
| Encoder vs Decoder | Choose right model for your task |
| Causal masking | Why GPT can generate text |
| LoRA fine-tuning | How to adapt LLMs efficiently |
| Temperature & sampling | How to control generation |
| RLHF | How ChatGPT/Claude are aligned |
