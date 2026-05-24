# 📝 Basics of NLP — Complete Guide for Gen AI

> **Why NLP?** Natural Language Processing is the field that teaches computers to understand, generate, and reason about human language. Gen AI is the culmination of decades of NLP research — knowing the foundations helps you understand why modern models work and where they came from.

---

## 1. What is NLP?

Natural Language Processing (NLP) is the intersection of **linguistics, computer science, and machine learning** — focused on enabling machines to process and understand text and speech.

```
Raw Text → [NLP Pipeline] → Insight / Action / Generated Text
```

### Core Challenges in NLP

| Challenge | Example |
|---|---|
| Ambiguity | "I saw a man with a telescope" — who has the telescope? |
| Context dependence | "It" can refer to different things depending on context |
| Idioms | "Break a leg" ≠ literally breaking a leg |
| Sarcasm | "Oh great, another bug" — opposite of literal meaning |
| Multiple languages | 7000+ languages in the world |

---

## 2. Text Preprocessing

Before feeding text to any ML model, it needs to be cleaned and normalized.

### Lowercasing
```python
text = "Hello World"
text.lower()   # "hello world"

# Important: Sometimes you DON'T want this (e.g., "US" vs "us")
```

### Removing Punctuation & Special Characters
```python
import re

text = "Hello, world! This is NLP #2025"
clean = re.sub(r'[^a-zA-Z0-9\s]', '', text)
# "Hello world This is NLP 2025"
```

### Tokenization
Splitting text into individual units (tokens).

```python
# Word tokenization
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

text = "NLP is amazing. Transformers changed everything!"
word_tokenize(text)
# ['NLP', 'is', 'amazing', '.', 'Transformers', 'changed', 'everything', '!']

sent_tokenize(text)
# ['NLP is amazing.', 'Transformers changed everything!']

# Simple split (naive)
text.split()
# ['NLP', 'is', 'amazing.', ...]
```

### Stop Word Removal
Remove common words that carry little semantic meaning.

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
# {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', ...}

tokens = word_tokenize("The cat sat on the mat")
filtered = [w for w in tokens if w.lower() not in stop_words]
# ['cat', 'sat', 'mat']

# Note: For modern LLMs, stop words are NOT removed — models learn context
```

### Stemming vs Lemmatization

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "better", "studies"]

# Stemming — crude, chops off suffix
[stemmer.stem(w) for w in words]
# ['run', 'run', 'ran', 'better', 'studi']   ← "studi" is wrong

# Lemmatization — linguistically correct, uses grammar rules
[lemmatizer.lemmatize(w, pos='v') for w in words]
# ['run', 'run', 'run', 'better', 'study']   ← correct
```

| | Stemming | Lemmatization |
|---|---|---|
| Speed | Fast | Slower |
| Accuracy | Lower | Higher |
| Output | Root fragment | Valid word |
| Use case | Search, IR | NLP analysis |

### Regex for Text Cleaning
```python
import re

# Remove URLs
re.sub(r'http\S+', '', text)

# Remove HTML tags
re.sub(r'<.*?>', '', html_text)

# Remove extra whitespace
re.sub(r'\s+', ' ', text).strip()

# Extract email addresses
re.findall(r'[\w.-]+@[\w.-]+\.\w+', text)

# Extract hashtags
re.findall(r'#\w+', tweet)
```

---

## 3. Text Representation — Turning Text into Numbers

### One-Hot Encoding
Each word gets a vector of all zeros except one position.

```
Vocab = ["cat", "dog", "mat"]
"cat" → [1, 0, 0]
"dog" → [0, 1, 0]
"mat" → [0, 0, 1]

Problem: High-dimensional, sparse, no semantic relationship
         "cat" and "dog" are equally distant as "cat" and "the"
```

### Bag of Words (BoW)
Represent a document by its **word frequencies** — ignoring order.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love NLP",
    "NLP is amazing",
    "I love machine learning and NLP"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
# ['amazing', 'and', 'i', 'is', 'learning', 'love', 'machine', 'nlp']

print(X.toarray())
# [[0, 0, 1, 0, 0, 1, 0, 1]   ← "I love NLP"
#  [1, 0, 0, 1, 0, 0, 0, 1]   ← "NLP is amazing"
#  [0, 1, 1, 0, 1, 1, 1, 1]]  ← "I love machine learning and NLP"
```

**Limitation:** Ignores word order. "Dog bites man" = "Man bites dog" in BoW.

### TF-IDF — Term Frequency × Inverse Document Frequency

Words that appear often in a document but rarely in others are important.

```
TF(word, doc)  = count of word in doc / total words in doc
IDF(word)      = log(total docs / docs containing word)
TF-IDF         = TF × IDF

Common words ("the", "is") → low IDF (appear in all docs)
Rare but relevant words → high IDF
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(corpus)
```

### N-grams — Capturing Word Order Partially

```python
# Unigrams: ["NLP", "is", "great"]
# Bigrams:  ["NLP is", "is great"]
# Trigrams: ["NLP is great"]

vectorizer = CountVectorizer(ngram_range=(1, 2))   # unigrams + bigrams
```

---

## 4. Word Embeddings — Semantic Representations

Word embeddings map words to **dense vectors** where similar words are close in vector space.

```
"king"  → [0.50, 0.12, -0.34, ...]
"queen" → [0.49, 0.13, -0.32, ...]   ← close to king
"cat"   → [-0.8, 0.91, 0.55, ...]    ← far from king

Famous analogy:
king - man + woman ≈ queen
```

### Word2Vec (2013, Google)

Two training approaches:

```
CBOW (Continuous Bag of Words):
  Context words → Predict center word
  Input: ["The", "_", "sat"] → Predict: "cat"

Skip-gram:
  Center word → Predict context words
  Input: "cat" → Predict: ["The", "sat"]
```

```python
from gensim.models import Word2Vec

sentences = [["the", "cat", "sat"], ["the", "dog", "ran"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, epochs=10)

# Get vector
model.wv["cat"]   # 100-dimensional vector

# Find similar words
model.wv.most_similar("cat")
# [('dog', 0.95), ('pet', 0.87), ...]

# Analogy
model.wv.most_similar(positive=["king", "woman"], negative=["man"])
# [('queen', 0.91), ...]
```

### GloVe — Global Vectors
Trained on co-occurrence statistics from the entire corpus.

```python
import numpy as np

# Load pre-trained GloVe vectors
glove = {}
with open("glove.6B.100d.txt") as f:
    for line in f:
        values = line.split()
        glove[values[0]] = np.array(values[1:], dtype='float32')

glove["cat"]   # 100-dimensional vector
```

### FastText
- Extension of Word2Vec
- Represents words as bag of **character n-grams**
- Handles **out-of-vocabulary (OOV) words** — can represent "unhappiness" even if not in training data

```python
from gensim.models import FastText
model = FastText(sentences, vector_size=100, window=5, min_count=1)
model.wv["unfamiliarword"]   # still works via character n-grams
```

### Comparing Embedding Methods

| Method | Dimension | Context? | OOV? | Year |
|---|---|---|---|---|
| One-Hot | Vocab size | ❌ | ❌ | — |
| Word2Vec | 100-300 | ❌ (static) | ❌ | 2013 |
| GloVe | 50-300 | ❌ (static) | ❌ | 2014 |
| FastText | 100-300 | ❌ (static) | ✅ | 2016 |
| BERT/GPT | 768-12288 | ✅ (dynamic) | ✅ | 2018+ |

> 🔑 **Static vs Contextual:** Word2Vec gives "bank" the same vector always. BERT gives "bank" different vectors in "river bank" vs "bank account" — **context-dependent**.

---

## 5. Classic NLP Tasks

### Text Classification
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("This movie is absolutely fantastic!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Sentiment Analysis
Classify text as Positive / Negative / Neutral.

```python
from textblob import TextBlob

blob = TextBlob("Transformers are absolutely incredible!")
blob.sentiment
# Sentiment(polarity=0.75, subjectivity=0.75)
# polarity: -1 (negative) to +1 (positive)
```

### Named Entity Recognition (NER)
Identify real-world entities in text.

```python
from transformers import pipeline

ner = pipeline("ner", aggregation_strategy="simple")
ner("Elon Musk founded SpaceX in California in 2002.")
# [{'entity_group': 'PER', 'word': 'Elon Musk'},
#  {'entity_group': 'ORG', 'word': 'SpaceX'},
#  {'entity_group': 'LOC', 'word': 'California'},
#  {'entity_group': 'DATE', 'word': '2002'}]
```

### Part-of-Speech (POS) Tagging
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog")
for token in doc:
    print(token.text, token.pos_, token.dep_)
# The     DET   det
# quick   ADJ   amod
# brown   ADJ   amod
# fox     NOUN  nsubj
# jumps   VERB  ROOT
```

### Machine Translation
```python
translator = pipeline("translation_en_to_fr")
translator("Hello, how are you?")
# [{'translation_text': 'Bonjour, comment allez-vous?'}]
```

### Text Summarization
```python
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer(long_article, max_length=130, min_length=30)
```

### Question Answering
```python
qa = pipeline("question-answering")
qa(question="Who invented transformers?",
   context="The transformer architecture was introduced by Vaswani et al. in 2017 at Google Brain.")
# {'answer': 'Vaswani et al.', 'score': 0.98}
```

---

## 6. Language Modeling

### N-gram Language Model
```
P("cat" | "the", "fat") = Count("the fat cat") / Count("the fat")

Limitations:
- Sparse: most n-gram sequences never seen in training
- No long-range dependencies
- Vocabulary coverage issues
```

### Neural Language Models
```
Input: "The cat sat on the"
Model: Neural Network (RNN, Transformer)
Output: Probability distribution over all vocab words
        {"mat": 0.42, "floor": 0.15, "bed": 0.08, ...}
Next token: sample from distribution
```

### Perplexity — Language Model Evaluation

```
Perplexity = exp(cross-entropy loss)

Lower perplexity = better model
A perplexity of 50 means: "as confused as uniformly guessing among 50 words"

Good LM perplexity on Wikipedia: ~15-30
Human-level perplexity: ~12-20
```

---

## 7. HuggingFace Ecosystem

The standard library for NLP and Gen AI in Python.

### Loading Models
```python
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Tokenizer + Model for any architecture
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# For text generation
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

### Running Inference
```python
import torch

# Tokenize input
inputs = tokenizer("Hello, I am a language model", return_tensors="pt")
# {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# For BERT: outputs.last_hidden_state → (batch, seq_len, 768)
# For GPT:  outputs.logits → (batch, seq_len, vocab_size)
```

### Generating Text
```python
inputs = tokenizer("Once upon a time", return_tensors="pt")

generated = model.generate(
    inputs["input_ids"],
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.2
)

tokenizer.decode(generated[0], skip_special_tokens=True)
```

### Pipeline API (Quick Use)
```python
from transformers import pipeline

# All major NLP tasks in one line
tasks = {
    "text-generation": pipeline("text-generation", model="gpt2"),
    "fill-mask": pipeline("fill-mask", model="bert-base-uncased"),
    "summarization": pipeline("summarization"),
    "translation": pipeline("translation_en_to_de"),
    "sentiment": pipeline("sentiment-analysis"),
    "ner": pipeline("ner"),
    "qa": pipeline("question-answering"),
    "zero-shot": pipeline("zero-shot-classification"),
}
```

### HuggingFace Datasets
```python
from datasets import load_dataset

# Load standard benchmarks
dataset = load_dataset("imdb")             # sentiment analysis
dataset = load_dataset("squad")            # question answering
dataset = load_dataset("wmt16", "de-en")   # translation

print(dataset["train"][0])
# {'text': 'A wonderful film...', 'label': 1}
```

---

## 8. Sentence Embeddings & Semantic Search

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat is on the mat.",
    "A kitten sits on a rug.",
    "The weather is nice today."
]

embeddings = model.encode(sentences)
# shape: (3, 384)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(embeddings)

# "cat on mat" vs "kitten on rug" → ~0.85 (high)
# "cat on mat" vs "weather" → ~0.12 (low)
```

### Use Case: Semantic Search (RAG)
```python
# Build knowledge base
docs = ["Document 1...", "Document 2...", "Document 3..."]
doc_embeddings = model.encode(docs)

# Query
query = "What is machine learning?"
query_embedding = model.encode([query])

# Find most similar document
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
best_doc = docs[similarities.argmax()]
```

---

## 9. Evaluation Metrics for NLP

### BLEU Score (Translation)
```
Measures n-gram overlap between generated and reference text.
Range: 0 to 1 (higher = better)
BLEU-4 > 0.3 considered good for translation
```

### ROUGE Score (Summarization)
```
ROUGE-N: n-gram recall between generated and reference
ROUGE-L: Longest common subsequence

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
scores = scorer.score(reference, generated)
```

### BERTScore
```
Computes similarity using BERT contextual embeddings
More semantic than n-gram metrics
```

### Perplexity (Language Models)
```python
import torch
import torch.nn.functional as F

def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()
```

---

## 10. Vector Databases — NLP Meets Gen AI

In Retrieval-Augmented Generation (RAG), we store document embeddings in a vector database and retrieve relevant context before generating.

```python
import faiss
import numpy as np

# Build index
dimension = 384
index = faiss.IndexFlatIP(dimension)   # inner product (= cosine if normalized)
index.add(doc_embeddings.astype('float32'))

# Search
query_vec = model.encode(["What is deep learning?"]).astype('float32')
distances, indices = index.search(query_vec, k=5)   # top 5 results

# Retrieved documents
relevant_docs = [docs[i] for i in indices[0]]
```

**Other vector databases:** Pinecone, Weaviate, Chroma, Qdrant, Milvus

---

## ✅ NLP Checklist for Gen AI

| Concept | Why It Matters for Gen AI |
|---|---|
| Tokenization | First step in every LLM pipeline |
| Text preprocessing | Data cleaning for fine-tuning datasets |
| TF-IDF | Keyword extraction, baseline retrieval |
| Word2Vec / GloVe | Historical embeddings, understand representation |
| Sentence embeddings | Power RAG systems and semantic search |
| Text classification | Fine-tuning task on downstream tasks |
| NER | Information extraction from LLM outputs |
| Language modeling & perplexity | Core evaluation metric for LLMs |
| HuggingFace pipeline | Standard tool for all Gen AI work |
| Vector databases (FAISS) | Essential for RAG and LLM memory |
| BLEU / ROUGE | Evaluation of generated text quality |
