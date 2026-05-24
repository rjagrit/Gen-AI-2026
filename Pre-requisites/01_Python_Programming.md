# 🐍 Python Programming — Complete Guide for Gen AI

> **Why Python?** Python is the #1 language for AI/ML. Nearly every Gen AI framework — PyTorch, HuggingFace, LangChain — is Python-first. Mastering Python is the first step in your journey.

---

## 1. Variables & Data Types

Variables store data. Python is **dynamically typed** — you don't declare types explicitly.

```python
name = "GPT"           # str
version = 4            # int
score = 98.6           # float
is_trained = True      # bool
nothing = None         # NoneType
```

### Common Type Operations
```python
type(name)             # <class 'str'>
isinstance(score, float)  # True
int("42")              # 42  — type casting
str(100)               # "100"
float("3.14")          # 3.14
```

---

## 2. Strings & Manipulation

Strings are sequences of characters — essential for NLP and prompt engineering.

```python
text = "Hello, Generative AI!"

# Basic operations
text.lower()           # "hello, generative ai!"
text.upper()           # "HELLO, GENERATIVE AI!"
text.strip()           # removes leading/trailing spaces
text.split(", ")       # ['Hello', 'Generative AI!']
" ".join(["a", "b"])   # "a b"
text.replace("AI", "Model")

# f-strings (most modern way)
model = "Claude"
print(f"My model is {model}")   # My model is Claude

# Slicing
text[0:5]              # "Hello"
text[-3:]              # "AI!"
```

---

## 3. Control Flow

### Conditionals
```python
temperature = 0.7

if temperature < 0.3:
    print("Very deterministic")
elif temperature < 0.7:
    print("Balanced")
else:
    print("Creative / Random")
```

### Loops
```python
# for loop
tokens = ["Hello", "world", "!"]
for token in tokens:
    print(token)

# while loop
count = 0
while count < 5:
    count += 1

# break / continue
for i in range(10):
    if i == 5:
        break       # stop loop
    if i % 2 == 0:
        continue    # skip even numbers
    print(i)
```

### List Comprehensions
```python
# Traditional loop
squares = []
for x in range(10):
    squares.append(x**2)

# Comprehension — cleaner and faster
squares = [x**2 for x in range(10)]

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Dict comprehension
word_lengths = {word: len(word) for word in ["cat", "elephant"]}
```

---

## 4. Data Structures

### Lists — Ordered, Mutable
```python
models = ["BERT", "GPT", "T5"]
models.append("LLaMA")        # add to end
models.insert(0, "Word2Vec")  # insert at index
models.pop()                  # remove last
models[1]                     # "GPT" — indexing
models[-1]                    # last element
models[1:3]                   # slicing
len(models)                   # length
sorted(models)                # alphabetical
```

### Tuples — Ordered, Immutable
```python
config = ("gpt-4", 8192, 0.7)   # model, context_len, temperature
model_name, ctx, temp = config   # unpacking
```

### Dictionaries — Key-Value Pairs
```python
model_info = {
    "name": "GPT-4",
    "params": "1.7T",
    "context": 128000
}

model_info["name"]             # "GPT-4"
model_info.get("version", "unknown")  # safe access with default
model_info.keys()              # dict_keys(['name', 'params', 'context'])
model_info.values()
model_info.items()             # key-value pairs

# Add / update
model_info["company"] = "OpenAI"
```

### Sets — Unique Elements
```python
vocab = {"the", "cat", "sat", "the"}   # {"the", "cat", "sat"}
vocab.add("mat")
vocab1 & vocab2    # intersection
vocab1 | vocab2    # union
vocab1 - vocab2    # difference
```

---

## 5. Functions

### Basic Functions
```python
def tokenize(text, lowercase=True):
    """Split text into word tokens."""
    if lowercase:
        text = text.lower()
    return text.split()

tokenize("Hello World")          # ['hello', 'world']
tokenize("Hello World", False)   # ['Hello', 'World']
```

### *args and **kwargs
```python
def log(*args, **kwargs):
    for arg in args:
        print(arg)
    for key, value in kwargs.items():
        print(f"{key} = {value}")

log("training", "started", epoch=1, loss=2.34)
```

### Lambda Functions
```python
# Anonymous one-line function
double = lambda x: x * 2
sort_by_len = sorted(["cat", "elephant", "ox"], key=lambda w: len(w))
```

### Recursion
```python
def factorial(n):
    if n <= 1:       # base case
        return 1
    return n * factorial(n - 1)   # recursive call
```

---

## 6. Object-Oriented Programming (OOP)

OOP is critical for understanding PyTorch models — every model is a class.

```python
class LanguageModel:
    """Base class for language models."""

    model_count = 0   # class variable

    def __init__(self, name, params_billions):
        self.name = name                      # instance variable
        self.params = params_billions
        LanguageModel.model_count += 1

    def describe(self):
        return f"{self.name} has {self.params}B parameters"

    def __repr__(self):
        return f"LanguageModel({self.name})"


class ChatModel(LanguageModel):
    """Inherits from LanguageModel."""

    def __init__(self, name, params, supports_tools=False):
        super().__init__(name, params)        # call parent __init__
        self.supports_tools = supports_tools

    def describe(self):                       # method overriding
        base = super().describe()
        return f"{base} | Tool use: {self.supports_tools}"


gpt4 = ChatModel("GPT-4", 1700, supports_tools=True)
print(gpt4.describe())
# GPT-4 has 1700B parameters | Tool use: True
```

---

## 7. File Handling

```python
# Write to file
with open("output.txt", "w") as f:
    f.write("Training complete\n")

# Read from file
with open("output.txt", "r") as f:
    content = f.read()
    # or line by line:
    for line in f:
        print(line.strip())

# JSON (very common in AI pipelines)
import json

data = {"model": "gpt-4", "tokens": 1024}
with open("config.json", "w") as f:
    json.dump(data, f, indent=2)

with open("config.json", "r") as f:
    config = json.load(f)

# CSV
import csv
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)
```

---

## 8. Exception Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
except (TypeError, ValueError) as e:
    print(f"Type/Value Error: {e}")
else:
    print("No error occurred")      # runs if no exception
finally:
    print("Always runs")            # cleanup code

# Raising exceptions
def load_model(path):
    if not path.endswith(".pt"):
        raise ValueError(f"Invalid model path: {path}")
```

---

## 9. Iterators & Generators

Generators are memory-efficient — crucial when processing huge datasets.

```python
# Regular function — builds full list in memory
def get_squares(n):
    return [x**2 for x in range(n)]

# Generator — yields one at a time
def generate_squares(n):
    for x in range(n):
        yield x**2    # pauses and returns value

gen = generate_squares(1_000_000)   # no memory used yet
next(gen)   # 0
next(gen)   # 1

# Generator expression
squares = (x**2 for x in range(10))

# Useful for streaming large text datasets
def stream_corpus(filepath):
    with open(filepath) as f:
        for line in f:
            yield line.strip()
```

---

## 10. NumPy — Numerical Computing

The backbone of ML math in Python.

```python
import numpy as np

# Creating arrays
a = np.array([1, 2, 3])           # 1D
b = np.array([[1, 2], [3, 4]])    # 2D matrix
np.zeros((3, 4))                   # 3x4 matrix of zeros
np.ones((2, 3))
np.eye(4)                          # 4x4 identity matrix
np.random.randn(3, 3)              # random normal values

# Shape & Reshape
a.shape        # (3,)
b.shape        # (2, 2)
b.reshape(4)   # [1, 2, 3, 4]

# Math operations (element-wise by default)
a + 10         # [11, 12, 13]
a * 2          # [2, 4, 6]
a ** 2         # [1, 4, 9]

# Matrix multiplication (critical for transformers)
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = A @ B           # or np.dot(A, B) — result shape (3, 5)

# Statistics
np.mean(a)     # 2.0
np.std(a)      # 0.816
np.max(a)      # 3
np.argmax(a)   # index of max = 2

# Broadcasting
matrix = np.ones((3, 3))
matrix + np.array([1, 2, 3])    # adds row-wise (broadcasting)
```

---

## 11. Pandas — Data Manipulation

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    "model": ["BERT", "GPT-2", "T5"],
    "params_M": [110, 1500, 11000],
    "year": [2018, 2019, 2020]
})

# Exploring data
df.head()              # first 5 rows
df.shape               # (3, 3)
df.dtypes              # column types
df.describe()          # statistics

# Accessing
df["model"]            # column as Series
df[df["year"] > 2018]  # filter rows
df.loc[0]              # row by label
df.iloc[0]             # row by index

# Operations
df["params_B"] = df["params_M"] / 1000
df.sort_values("params_M", ascending=False)
df.dropna()            # drop missing rows
df.groupby("year")["params_M"].mean()
```

---

## 12. Modules & Packages

```python
# Import standard library
import os
import sys
import math
from pathlib import Path
from collections import defaultdict, Counter

# Installing packages
# pip install torch transformers datasets

# Virtual environments (isolate project dependencies)
# python -m venv myenv
# source myenv/bin/activate  (Linux/Mac)
# myenv\Scripts\activate     (Windows)
```

---

## ✅ Python Concepts Checklist for Gen AI

| Concept | Importance |
|---|---|
| Lists, Dicts, Sets | ⭐⭐⭐ Essential |
| Functions & Lambda | ⭐⭐⭐ Essential |
| Classes & OOP | ⭐⭐⭐ Essential (PyTorch models are classes) |
| NumPy arrays & math | ⭐⭐⭐ Essential |
| Pandas DataFrames | ⭐⭐ Important |
| Generators / yield | ⭐⭐ Important (large datasets) |
| File I/O & JSON | ⭐⭐ Important |
| Exception handling | ⭐ Helpful |
| List comprehensions | ⭐⭐ Important |
