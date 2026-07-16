# 🎓 USU Computer Science Undergraduate Thesis Guide (RAG)

An LLM-powered question-answering system built with **Retrieval-Augmented Generation (RAG)** to help Computer Science students at **Universitas Sumatera Utara (USU)** understand and navigate the official undergraduate thesis (*Skripsi S-1*) guidelines.

Instead of manually digging through a long, dense, and often image-heavy guideline document, students can simply ask a question in natural language — e.g. *"What is the maximum number of pages for Chapter 1?"* or *"What citation format should I use?"* — and get an instant, grounded answer sourced directly from the official guideline.

---

## ✨ Features

- **Natural language Q&A** over the official Fasilkom-TI USU thesis writing guideline
- **Semantic chunking** of the source document for more contextually coherent retrieval
- **Vector search** powered by Qdrant for fast and accurate similarity search
- **Reranking** with CrossEncoder to improve the relevance of retrieved context before generation
- **RAG evaluation pipeline** using RAGAS to measure faithfulness, relevancy, and overall answer quality
- **Web-based chat interface** for an interactive, conversational experience

## Website
**Main Page**
<img width="1910" height="915" alt="Main Page" src="https://github.com/user-attachments/assets/80c58bed-b3a8-4f93-9d21-20afc5a9c1a7" />
**Chat Page**
<img width="1910" height="1503" alt="image" src="https://github.com/user-attachments/assets/c45f096c-5b2d-465b-b6ac-ece43691c4a7" />


## 🧠 How It Works

The pipeline is split into two main stages:

1. **Document Preparation**
   - The official thesis guideline (originally a PDF/DOCX file containing both text and rasterized/image-embedded content) is converted into clean, structured Markdown.
   - The document is then split into semantically meaningful chunks rather than naive fixed-size splits, preserving the context of each rule/section.

2. **Retrieval-Augmented Generation**
   - Each chunk is embedded using a sentence-transformer model and stored in a **Qdrant** vector database.
   - When a user asks a question, the system retrieves the most relevant chunks, reranks them with **FlashRank**, and feeds the refined context to an LLM to generate a grounded, accurate answer.
   - The end-to-end pipeline is evaluated using **RAGAS** metrics to ensure answer quality stays high as the system evolves.

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python, Jupyter Notebook |
| RAG Orchestration | LangChain (`langchain-community`, `langchain_huggingface`) |
| Embeddings | `sentence-transformers` |
| Chunking | `semantic-text-splitter` |
| Vector Database | Qdrant (`qdrant-client`) |
| Model | llama-3.1-8b-instruct |
| Reranking | CrossEncoder|
| Evaluation | RAGAS |
| Web App | Flask (HTML/CSS/JS frontend) |
| Deployment | Vercel |

## 📊 Evaluation

The RAG pipeline is evaluated using **RAGAS**, with results stored in `ragas_hasil.csv`. Metrics tracked include answer faithfulness, context relevancy, and overall answer correctness against the official guideline — helping validate that the system's answers stay grounded in the source document rather than hallucinated.
- **faithfulness** = 76%
- **answer_relevancy** = 81%
- **context_precision** = 56%
- **context_recall** = 76%
- **answer_correctness** = 64%

## 🎯 Motivation

USU thesis guidelines are often long, formatted as scanned/rasterized pages, and tedious to search through manually. This project aims to make that information instantly accessible to students, reducing friction in one of the more confusing parts of the thesis-writing process.
