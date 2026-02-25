# 🤖 Research Augmentation Agent

**Autonomous Multi-Agent Scientific Literature Analysis**

> Free Stack: HuggingFace + LangChain v0.3+ + ChromaDB + Streamlit

---

## 🚀 Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "first commit"
git remote add origin https://github.com/YOUR_USERNAME/research-agent.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repo
4. Main file: `app.py`
5. Click **Deploy** ✅

---

## 🗂 Project Structure
```
research-agent/
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Dependencies
├── .streamlit/
│   └── config.toml         ← Theme & server config
└── README.md
```

---

## ⚙️ Tech Stack

| Tool | Role | Free? |
|------|------|-------|
| Mistral-7B (HuggingFace) | LLM | ✅ |
| BGE-small | Embeddings | ✅ |
| ChromaDB | Vector DB | ✅ |
| LangChain v0.3+ | RAG pipeline | ✅ |
| arXiv API | Paper search | ✅ |
| ReportLab | PDF export | ✅ |
| Streamlit | UI | ✅ |

---

## 🔄 Pipeline

```
Query → arXiv Search → PDF Processing → ChromaDB Embedding
      → RAG Analysis (Mistral-7B) → Report Generation → PDF Export
```

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
