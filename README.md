---
title: Research Augmentation Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🤖 Autonomous Research Augmentation Agent

**Free Stack:** HuggingFace + LangChain + ChromaDB + Gradio

## Features
- 🔍 Search arXiv automatically
- 📄 Process & chunk PDFs
- 🧠 ChromaDB vector storage
- ⚡ RAG with Mistral-7B (free)
- 📊 Structured report generation
- 📄 Export to PDF

## How to Use
1. Enter your research query
2. Click **Run Pipeline**
3. Watch the 4 tabs fill up:
   - **Papers Found** — retrieved from arXiv
   - **RAG Analysis** — AI analysis
   - **Full Report** — structured markdown report
   - **Download PDF** — export your report

## Tech Stack
| Tool | Role | Cost |
|------|------|------|
| Mistral-7B | LLM via HuggingFace | Free |
| BGE-small | Embeddings | Free |
| ChromaDB | Vector DB | Free |
| LangChain | RAG pipeline | Free |
| arXiv API | Paper search | Free |
| Gradio | UI | Free |
