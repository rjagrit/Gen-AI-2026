@REM ============================================================
@REM   GEN AI - COMPLETE PREREQUISITES CHEATSHEET
@REM   Topics: Python | ML | Deep Learning | Transformers | NLP
@REM ============================================================


@REM ════════════════════════════════════════════════════════════
@REM   1. PYTHON PROGRAMMING
@REM ════════════════════════════════════════════════════════════

@REM --- BASICS ---
@REM   Variables & Data Types      : int, float, str, bool, None
@REM   Operators                   : Arithmetic, Comparison, Logical, Bitwise
@REM   String Manipulation         : f-strings, .split(), .join(), .strip(), .format()
@REM   Type Casting                : int(), float(), str(), list()
@REM   Input / Output              : print(), input()

@REM --- CONTROL FLOW ---
@REM   Conditionals                : if / elif / else
@REM   Loops                       : for, while, break, continue, pass
@REM   Comprehensions              : List, Dict, Set comprehensions

@REM --- DATA STRUCTURES ---
@REM   Lists                       : Indexing, slicing, append, pop, sort
@REM   Tuples                      : Immutable sequences
@REM   Dictionaries                : key-value pairs, .get(), .items(), .keys()
@REM   Sets                        : Unique elements, union, intersection
@REM   Stacks & Queues             : Using lists and collections.deque

@REM --- FUNCTIONS ---
@REM   def, return, *args, **kwargs
@REM   Lambda functions            : lambda x: x * 2
@REM   Scope                       : local, global, nonlocal
@REM   Recursion                   : Base case + recursive call

@REM --- OOP (Object-Oriented Programming) ---
@REM   Classes & Objects           : class MyClass:
@REM   __init__ constructor        : self keyword
@REM   Inheritance                 : class Child(Parent):
@REM   Encapsulation               : private attributes _ and __
@REM   Polymorphism                : method overriding
@REM   Magic Methods               : __str__, __len__, __repr__

@REM --- FILE HANDLING ---
@REM   Read / Write files          : open(), read(), write(), with statement
@REM   JSON                        : json.load(), json.dump()
@REM   CSV                         : csv.reader(), csv.writer()

@REM --- IMPORTANT LIBRARIES ---
@REM   NumPy                       : Arrays, matrix ops, broadcasting
@REM     np.array(), np.zeros(), np.ones(), np.reshape(), np.dot()
@REM     np.mean(), np.std(), np.random.randn()
@REM
@REM   Pandas                      : DataFrames, Series
@REM     pd.read_csv(), df.head(), df.describe(), df.dropna()
@REM     df.groupby(), df.merge(), df.apply()
@REM
@REM   Matplotlib / Seaborn        : Plotting & visualization
@REM     plt.plot(), plt.scatter(), plt.hist(), sns.heatmap()

@REM --- EXCEPTION HANDLING ---
@REM   try / except / finally / raise
@REM   Common exceptions: ValueError, TypeError, IndexError, KeyError

@REM --- ITERATORS & GENERATORS ---
@REM   iter(), next(), yield keyword
@REM   Generator expressions       : (x**2 for x in range(10))

@REM --- MODULES & PACKAGES ---
@REM   import, from ... import, as
@REM   pip install, virtual environments (venv)


@REM ════════════════════════════════════════════════════════════
@REM   2. MACHINE LEARNING FUNDAMENTALS
@REM ════════════════════════════════════════════════════════════

@REM --- CORE CONCEPTS ---
@REM   What is ML?                 : Learning patterns from data
@REM   Types of ML:
@REM     Supervised Learning       : Labeled data (Classification, Regression)
@REM     Unsupervised Learning     : Unlabeled data (Clustering, Dimensionality Reduction)
@REM     Reinforcement Learning    : Agent, environment, reward
@REM     Self-Supervised Learning  : Labels derived from data itself (key for Gen AI)

@REM --- SUPERVISED LEARNING ALGORITHMS ---
@REM   Linear Regression           : Predict continuous values
@REM   Logistic Regression         : Binary classification (sigmoid output)
@REM   Decision Trees              : Rule-based splits on features
@REM   Random Forest               : Ensemble of decision trees
@REM   Support Vector Machines     : Hyperplane-based classification
@REM   K-Nearest Neighbors (KNN)   : Classify by nearest data points
@REM   Naive Bayes                 : Probabilistic classifier

@REM --- UNSUPERVISED LEARNING ---
@REM   K-Means Clustering          : Group data into K clusters
@REM   Hierarchical Clustering     : Tree-based cluster structure
@REM   PCA (Principal Component Analysis) : Reduce dimensions, retain variance
@REM   t-SNE / UMAP               : 2D/3D visualization of high-dim data

@REM --- CORE ML MATH ---
@REM   Linear Algebra              : Vectors, Matrices, Dot product, Transpose, Inverse
@REM   Calculus                    : Derivatives, Chain rule, Gradients
@REM   Probability                 : Bayes theorem, Distributions (Normal, Bernoulli)
@REM   Statistics                  : Mean, Median, Variance, Std dev, Correlation

@REM --- TRAINING CONCEPTS ---
@REM   Features (X) & Labels (Y)
@REM   Train / Validation / Test Split
@REM   Overfitting & Underfitting
@REM   Bias-Variance Tradeoff
@REM   Cross-Validation            : k-Fold CV

@REM --- LOSS FUNCTIONS ---
@REM   MSE (Mean Squared Error)    : Regression
@REM   MAE (Mean Absolute Error)   : Regression
@REM   Cross-Entropy Loss          : Classification
@REM   Binary Cross-Entropy        : Binary classification

@REM --- OPTIMIZATION ---
@REM   Gradient Descent            : Update params to minimize loss
@REM   Stochastic GD (SGD)         : One sample at a time
@REM   Mini-Batch Gradient Descent : Most common in practice
@REM   Adam Optimizer              : Adaptive learning rate (default for Gen AI models)
@REM   Learning Rate               : Step size for weight update
@REM   Learning Rate Scheduling    : Decay, Warmup

@REM --- EVALUATION METRICS ---
@REM   Classification  : Accuracy, Precision, Recall, F1-Score, AUC-ROC
@REM   Regression      : MSE, RMSE, MAE, R-squared
@REM   Confusion Matrix

@REM --- SKLEARN ESSENTIALS ---
@REM   from sklearn.model_selection import train_test_split
@REM   from sklearn.preprocessing import StandardScaler, LabelEncoder
@REM   from sklearn.metrics import accuracy_score, classification_report
@REM   model.fit(X_train, y_train)
@REM   model.predict(X_test)


@REM ════════════════════════════════════════════════════════════
@REM   3. DEEP LEARNING CONCEPTS
@REM ════════════════════════════════════════════════════════════

@REM --- NEURAL NETWORK BASICS ---
@REM   Neuron / Perceptron         : Weighted sum + activation function
@REM   Layers                      : Input layer, Hidden layers, Output layer
@REM   Weights & Biases            : Parameters learned during training
@REM   Forward Pass                : Input -> prediction
@REM   Backward Pass               : Compute gradients (Backpropagation)
@REM   Epoch                       : One full pass over training data
@REM   Batch Size                  : Samples processed before weight update

@REM --- ACTIVATION FUNCTIONS ---
@REM   ReLU                        : max(0, x)  — most common hidden layer
@REM   Sigmoid                     : 1/(1+e^-x) — binary output
@REM   Softmax                     : Multi-class probability distribution
@REM   Tanh                        : Range (-1, 1)
@REM   GELU                        : Used in Transformers (Gen AI)
@REM   Leaky ReLU                  : Fixes dying ReLU problem

@REM --- BACKPROPAGATION ---
@REM   Chain Rule                  : Gradient flows backward through layers
@REM   Gradient of Loss w.r.t. Weights
@REM   Weight Update               : w = w - lr * gradient

@REM --- REGULARIZATION ---
@REM   Dropout                     : Randomly zero out neurons during training
@REM   L1 / L2 Regularization      : Penalize large weights
@REM   Batch Normalization         : Normalize layer inputs — speeds training
@REM   Layer Normalization         : Used in Transformers (not Batch Norm)
@REM   Early Stopping              : Stop when val loss stops improving

@REM --- CNN (Convolutional Neural Networks) ---
@REM   Convolution Operation       : Filter slides over input
@REM   Feature Maps                : Output of convolution
@REM   Pooling                     : MaxPool, AvgPool — downsampling
@REM   Padding & Stride
@REM   Used for                    : Image classification, object detection
@REM   Architectures               : LeNet, AlexNet, VGG, ResNet, EfficientNet

@REM --- RNN / LSTM / GRU (Sequence Models) ---
@REM   RNN                         : Recurrent connection for sequences
@REM   Vanishing Gradient Problem  : RNN limitation for long sequences
@REM   LSTM                        : Long Short-Term Memory — forget/input/output gates
@REM   GRU                         : Gated Recurrent Unit — simpler than LSTM
@REM   Bidirectional RNN           : Reads sequence both ways
@REM   Used for                    : Text, time-series (pre-transformer era)

@REM --- FRAMEWORKS ---
@REM   TensorFlow / Keras          : Google's DL framework
@REM   PyTorch                     : Facebook's DL framework (dominant in Gen AI)
@REM     torch.Tensor              : Core data structure
@REM     torch.nn.Module           : Base class for models
@REM     torch.optim               : Optimizers (Adam, SGD)
@REM     DataLoader, Dataset       : Efficient data loading
@REM     .to(device)               : Move tensors to GPU (CUDA)

@REM --- TRAINING PIPELINE ---
@REM   1. Load & preprocess data
@REM   2. Define model architecture
@REM   3. Set loss function & optimizer
@REM   4. Forward pass -> compute loss
@REM   5. Backward pass -> compute gradients
@REM   6. Update weights
@REM   7. Evaluate on validation set
@REM   8. Repeat for N epochs


@REM ════════════════════════════════════════════════════════════
@REM   4. UNDERSTANDING TRANSFORMERS (like GPT)
@REM ════════════════════════════════════════════════════════════

@REM --- WHY TRANSFORMERS? ---
@REM   RNNs fail on long sequences (vanishing gradient)
@REM   Transformers process ALL tokens in PARALLEL
@REM   "Attention is All You Need" — Vaswani et al., 2017

@REM --- TOKENIZATION ---
@REM   Token                       : Basic unit of text (word, subword, char)
@REM   Vocabulary                  : Set of all known tokens
@REM   Tokenizers                  : BPE, WordPiece, SentencePiece
@REM   Token IDs                   : Integer representation of tokens
@REM   [CLS], [SEP], [PAD], [MASK] : Special tokens used in BERT-style models

@REM --- EMBEDDINGS ---
@REM   Token Embeddings            : Vector representation of each token
@REM   Positional Encoding         : Injects position info (sin/cos functions)
@REM     — Transformers have no built-in order, need explicit position signal
@REM   Word2Vec, GloVe             : Older static embeddings (pre-transformer)
@REM   Contextual Embeddings       : Depend on surrounding context (transformers)

@REM --- SELF-ATTENTION (Core Mechanism) ---
@REM   Query (Q), Key (K), Value (V) : Learned projections of input
@REM   Attention Score             : softmax(QK^T / sqrt(d_k)) * V
@REM   sqrt(d_k) scaling           : Prevents vanishing gradients in softmax
@REM   What it does                : Each token attends to ALL other tokens
@REM   Attention Weights           : How much each token focuses on others

@REM --- MULTI-HEAD ATTENTION ---
@REM   Run H parallel attention heads
@REM   Each head learns different relationships
@REM   Concatenate + Linear projection
@REM   Captures diverse patterns in one layer

@REM --- TRANSFORMER BLOCK ---
@REM   1. Multi-Head Self-Attention
@REM   2. Add & Norm (Residual connection + LayerNorm)
@REM   3. Feed-Forward Network (FFN / MLP)
@REM   4. Add & Norm again
@REM   Stacked N times (e.g., GPT-3 has 96 layers)

@REM --- ENCODER vs DECODER ---
@REM   Encoder-only (BERT)         : Reads full sequence, good for understanding tasks
@REM   Decoder-only (GPT)          : Generates tokens left-to-right (autoregressive)
@REM   Encoder-Decoder (T5, BART)  : Seq2Seq — translation, summarization

@REM --- CAUSAL / MASKED ATTENTION ---
@REM   GPT uses causal masking     : Can only attend to LEFT context (past tokens)
@REM   Prevents future token leakage during training

@REM --- POSITIONAL EMBEDDINGS (Variants) ---
@REM   Sinusoidal                  : Original paper
@REM   Learned Positional          : Trainable (GPT-2)
@REM   RoPE (Rotary)               : Used in LLaMA, GPT-NeoX
@REM   ALiBi                       : Better length generalization

@REM --- KEY MODELS TO KNOW ---
@REM   BERT          : Encoder-only, bidirectional, masked LM pretraining
@REM   GPT-2/3/4     : Decoder-only, autoregressive generation
@REM   T5            : Encoder-Decoder, "text-to-text" framework
@REM   LLaMA 2/3     : Open-source GPT-style models
@REM   BART          : Encoder-Decoder for summarization/translation
@REM   RoBERTa       : Improved BERT training
@REM   Claude        : Constitutional AI, harmless + helpful

@REM --- PRETRAINING vs FINE-TUNING ---
@REM   Pretraining                 : Learn on massive text data (next-token prediction)
@REM   Fine-Tuning                 : Adapt to specific task with smaller dataset
@REM   PEFT                        : Parameter-Efficient Fine-Tuning
@REM     LoRA                      : Low-Rank Adaptation — tune small matrices
@REM     Adapters                  : Small bottleneck layers inserted in model
@REM   Instruction Tuning          : Fine-tune on instruction-response pairs
@REM   RLHF                        : Reinforcement Learning from Human Feedback (ChatGPT)

@REM --- INFERENCE CONCEPTS ---
@REM   Temperature                 : Controls randomness (0=greedy, >1=creative)
@REM   Top-k Sampling              : Sample from top k probable tokens
@REM   Top-p (Nucleus) Sampling    : Sample from tokens summing to probability p
@REM   Beam Search                 : Keep N best sequences at each step
@REM   Context Window              : Max tokens model can process at once


@REM ════════════════════════════════════════════════════════════
@REM   5. BASICS OF NLP (Natural Language Processing)
@REM ════════════════════════════════════════════════════════════

@REM --- TEXT PREPROCESSING ---
@REM   Lowercasing                 : Normalize text
@REM   Punctuation Removal         : Clean noise
@REM   Tokenization                : Split text into tokens
@REM   Stop Word Removal           : Remove "the", "is", "at" etc.
@REM   Stemming                    : Reduce to root form (run -> run)
@REM   Lemmatization               : Linguistically correct root (better than stemming)
@REM   Regex                       : Pattern matching for text cleaning

@REM --- TEXT REPRESENTATION ---
@REM   One-Hot Encoding            : Sparse binary vectors
@REM   Bag of Words (BoW)          : Word frequency counts
@REM   TF-IDF                      : Term Frequency * Inverse Document Frequency
@REM   N-grams                     : Contiguous sequences of N words/chars
@REM   Word Embeddings             : Dense vector representations
@REM     Word2Vec                  : CBOW and Skip-gram models
@REM     GloVe                     : Global Vectors — co-occurrence matrix
@REM     FastText                  : Subword-level embeddings

@REM --- CLASSIC NLP TASKS ---
@REM   Text Classification         : Spam detection, sentiment analysis
@REM   Named Entity Recognition (NER) : Identify people, places, dates
@REM   Part-of-Speech (POS) Tagging : Noun, Verb, Adjective labels
@REM   Dependency Parsing          : Grammatical structure of sentence
@REM   Coreference Resolution      : "he", "she" -> actual entity
@REM   Machine Translation         : English -> French etc.
@REM   Text Summarization          : Extractive vs Abstractive
@REM   Question Answering          : Extract or generate answers
@REM   Sentiment Analysis          : Positive / Negative / Neutral

@REM --- LANGUAGE MODELING ---
@REM   N-gram Language Model       : P(word | previous N-1 words)
@REM   Neural Language Model       : Use NN to predict next word
@REM   Perplexity                  : Evaluation metric — lower is better
@REM   Next Token Prediction       : Core objective of GPT models
@REM   Masked Language Modeling    : Core objective of BERT ([MASK] prediction)

@REM --- MODERN NLP PIPELINE ---
@REM   Raw Text
@REM     -> Tokenization (BPE/WordPiece)
@REM       -> Token IDs
@REM         -> Embeddings (+ Positional)
@REM           -> Transformer Encoder/Decoder
@REM             -> Task-specific Head (classification, generation, etc.)

@REM --- HUGGING FACE ECOSYSTEM ---
@REM   transformers library        : Pre-trained models (BERT, GPT, T5...)
@REM   from transformers import AutoTokenizer, AutoModelForCausalLM
@REM   pipeline()                  : Easy high-level API
@REM     pipeline("text-generation", model="gpt2")
@REM     pipeline("sentiment-analysis")
@REM     pipeline("summarization")
@REM   datasets library            : Load NLP benchmark datasets
@REM   tokenizers library          : Fast tokenization
@REM   PEFT library                : LoRA, Adapters
@REM   Accelerate                  : Multi-GPU / distributed training

@REM --- KEY NLP METRICS ---
@REM   BLEU Score                  : Translation quality (n-gram overlap)
@REM   ROUGE Score                 : Summarization quality (recall-based)
@REM   BERTScore                   : Semantic similarity using BERT embeddings
@REM   Perplexity                  : Language model fluency

@REM --- VECTOR DATABASES (for Gen AI RAG) ---
@REM   Embeddings stored in vector DB
@REM   Similarity Search           : Cosine similarity, dot product
@REM   Tools                       : FAISS, Pinecone, Weaviate, Chroma
@REM   Used in RAG                 : Retrieval-Augmented Generation


@REM ════════════════════════════════════════════════════════════
@REM   GEN AI SPECIFIC CONCEPTS (Bonus — What ties it all together)
@REM ════════════════════════════════════════════════════════════

@REM   Large Language Models (LLMs): GPT-4, Claude, LLaMA, Gemini
@REM   Prompt Engineering          : Zero-shot, Few-shot, Chain-of-Thought
@REM   RAG (Retrieval-Augmented Generation): LLM + external knowledge base
@REM   Agents & Tools              : LLM calls external APIs/functions
@REM   Diffusion Models            : Stable Diffusion, DALL-E (image generation)
@REM   Multimodal Models           : Text + Image (GPT-4V, Gemini)
@REM   Quantization                : INT8/INT4 to reduce model size
@REM   GGUF / GGML                 : Formats for running LLMs locally (Ollama)
@REM   LangChain / LlamaIndex      : Frameworks for building Gen AI apps
@REM   Evaluation                  : Faithfulness, Relevance, Hallucination rate


@REM ============================================================
@REM   RECOMMENDED LEARNING ORDER
@REM ============================================================
@REM   Step 1 : Python + NumPy + Pandas
@REM   Step 2 : ML Fundamentals (Sklearn)
@REM   Step 3 : Deep Learning (PyTorch basics)
@REM   Step 4 : NLP Basics + Text Preprocessing
@REM   Step 5 : Transformers (Attention mechanism, BERT, GPT)
@REM   Step 6 : Hugging Face ecosystem
@REM   Step 7 : Gen AI apps (RAG, Agents, Fine-tuning)
@REM ============================================================
