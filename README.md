#  Quantum Computing History RAG System

##  Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for answering questions based on a small, domain-specific dataset: **History of Quantum Computing**.

The system retrieves relevant information from a curated dataset and uses a local Large Language Model (LLM) to generate grounded answers.

---

## Objective

The goal of this project is to:

- Build a complete RAG pipeline from scratch
- Design a meaningful evaluation framework
- Analyze system performance and limitations

---

##  Dataset

The dataset consists of **5 domain-specific documents**, each focused on a key topic:

- Feynman's Proposal (1982)
- Shor's Algorithm (1994)
- Grover's Algorithm (1996)
- Quantum Computing Basics
- Timeline of Quantum Computing

Each document:
- Contains factual information (names, dates, definitions)
- Is written in structured paragraphs for better retrieval
- Supports answering 10–15 domain-specific questions

---

## RAG Pipeline Design

### 1. Chunking Strategy
- Sentence-window chunking
- Overlapping chunks (window=5, step=3)
- Preserves semantic context

---

### 2. Embeddings
- Model: `all-MiniLM-L6-v2` (Sentence-BERT)
- Dense semantic embeddings
- Fallback: TF-IDF (for restricted environments)

---

### 3. Retrieval Strategy (Hybrid)

The system uses **Hybrid Retrieval**:

- **BM25** → keyword-based relevance  
- **SBERT** → semantic similarity  

Final score:

Score = α * BM25 + (1 - α) * Dense Similarity


Adaptive weighting:
- Short queries → more weight to BM25  
- Long queries → more weight to semantic similarity  

---

### 4. Vector Store
- FAISS (`IndexFlatIP`)
- Cosine similarity (via normalization)
- Persistent index storage

---

### 5. Generation (LLM)

- Model: `google/flan-t5-base`
- Runs locally (no API required)
- Generates answers grounded in retrieved context

---
### 🔄 Full Pipeline Flow

- **User Query**
- **Embedding**
- **Hybrid Retrieval (BM25 + SBERT)**
- **Top-K Relevant Chunks**
- **Prompt Construction**
- **LLM (Flan-T5)**
- **Final Answer**

---

##  Evaluation Framework

The system includes a custom evaluation pipeline with multiple metrics:

### 1. Semantic Similarity
- Cosine similarity between predicted and ground truth answers

---

### 2. Keyword Overlap
- Measures shared important terms between answers

---

### 3. Length Score
- Penalizes overly short or incomplete answers

---

### 4. Retrieval Accuracy
- Checks if relevant information was retrieved

---

###  Final Score


Final Score =
0.4 × Semantic Similarity +
0.3 × Keyword Overlap +
0.2 × Length Score +
0.1 × Retrieval Score


---

##  Example Evaluation Result

| Metric | Score |
|------|------|
| Semantic Similarity | 0.88 |
| Keyword Overlap | 0.75 |
| Length Score | 0.90 |
| Retrieval Accuracy | 1.00 |
| **Final Score** | **0.86** |

---

##  Challenges & Learnings

### 1. LLM Limitations
- Small models (Flan-T5) often generate:
  - overly short answers  
  - or overly long responses  

### 2. Prompt Engineering
- Prompt design significantly impacts output quality
- Over-constraining prompts leads to poor responses

### 3. Retrieval vs Generation Tradeoff
- Accurate retrieval does not guarantee good answers
- Separation of retrieval and generation is critical

### 4. Hybrid Retrieval Benefits
- BM25 improves keyword matching
- SBERT improves semantic understanding
- Combined approach improves robustness

---

##  Key Insights

- **Data quality > Model size**
- **Retrieval quality directly impacts answer quality**
- **Evaluation is essential for understanding system behavior**

---
## 📦 Installation

```bash
pip install numpy scikit-learn sentence-transformers transformers torch faiss-cpu rank_bm25
```

**->>** Running the Project

```bash
python main.py
```

**-->** Running Evaluation

```bash
def evaluation/evaluate.py
```
**📁 Project Structure**
```
rag-project/
│
├── .venv/
│
├── data/
│   ├── doc1_feynman.txt
│   ├── doc2_shor.txt
│   ├── doc3_grover.txt
│   ├── doc4_quantum_basic.txt
│   └── doc5_timeline.txt
│
├── evaluation/
│   ├── evaluate.py
│   └── question.json
│
├── index/
│   ├── quantum_rag.chunks
│   └── quantum_rag.faiss
│
├── src/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── rag_pipeline.py
│   └── utils.py
│
├── main.py
├── README.md
└── requirements.txt
```
# Conclusion

This project demonstrates a complete RAG pipeline with:

- Hybrid retrieval
- Local LLM integration
- Custom evaluation framework

It highlights the importance of:

- Retrieval quality
- Prompt design
- Evaluation methodology