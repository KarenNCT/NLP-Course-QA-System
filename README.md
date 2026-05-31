# NLP Course QA System

## Project Description

A Retrieval-Augmented Generation (RAG) question answering system for NLP course materials.

Pipeline:

PDF
→ Text Extraction
→ Chunking
→ BGE-M3 Embedding
→ FAISS Retrieval
→ Qwen2.5-3B
→ Answer Generation

## Environment

Python 3.11

## Install

```bash
pip install -r requirements.txt
```

## Run

Build knowledge base:

```bash
python build_faiss.py
```

Question answering:

```bash
python QA.py
```

Input:

input.csv

Output:

output.csv