# 🎓 Academic Hybrid-RAG System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green?style=for-the-badge)
![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-purple?style=for-the-badge)

## 📌 Project Overview
The **Academic Hybrid-RAG System** is an advanced Retrieval-Augmented Generation pipeline designed specifically for university students and researchers. It ingests lecture PDFs, chunks them intelligently, and utilizes a state-of-the-art **Hybrid Search Engine** to retrieve context-aware answers to complex academic queries. The system enforces strict citation rules, eliminating AI hallucination by directly referencing source files and page numbers.

---

## 🌟 Key Features
* **🧠 Hybrid Retrieval Strategy:** Combines Lexical Search (`BM25`) for exact keyword matching and Semantic Search (`SentenceTransformers` + `FAISS`) for contextual understanding.
* **🔗 Reciprocal Rank Fusion (RRF):** Intelligently fuses search results from different retrieval algorithms to achieve state-of-the-art accuracy.
* **🛡️ Hallucination-Free Synthesis:** The LLM is strictly prompted to include inline citations (e.g., `[Source: Lecture.pdf | Page: 5]`) for every claim.
* **⚡ Blazing Fast Vector Search:** Utilizes FAISS `IndexFlatIP` for rapid inner-product vector similarity calculations.
* **📊 Performance Tracking:** Real-time display of retrieval and generation latency metrics.

---

## 📂 Project Structure

```text
lecture_retrieval_engine/
│
├── app.py                 # Main Streamlit application and UI
├── config.py              # Hyperparameters and environment variables
├── evaluator.py           # Precision, Recall, and MRR calculation metrics
├── indexer.py             # FAISS vector indexing implementation
├── pdf_parser.py          # PDF extraction, text cleaning, and chunking
├── retrievers.py          # BM25, Dense, and Hybrid RRF search engines
├── requirements.txt       # Project dependencies
├── .env.example           # Template for environment secrets
├── .gitignore             # Git ignore configuration
└── README.md              # Project documentation (You are here)