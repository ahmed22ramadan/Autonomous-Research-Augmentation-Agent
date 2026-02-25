"""
Autonomous Research Augmentation Agent
Stack: HuggingFace + LangChain v0.3+ + ChromaDB + Streamlit
Deploy: Streamlit Cloud (free) from GitHub repo
"""

import os, re, warnings, datetime, tempfile, requests
warnings.filterwarnings("ignore")

import arxiv
import fitz
import torch
import streamlit as st

# LangChain v0.3+ imports (no deprecation warnings)
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from transformers import (
    pipeline as hf_pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER

# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════
MODEL_NAME  = "mistralai/Mistral-7B-Instruct-v0.2"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE  = 512
CHUNK_OVER  = 64
TOP_K       = 6
USE_GPU     = torch.cuda.is_available()

# ══════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="Research Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
    background-color: #07091a !important;
    color: #dde0ff !important;
}
.stApp { background: #07091a; }
.stApp > header { background: transparent !important; }

/* Header */
.hero { text-align: center; padding: 28px 0 8px; }
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem; font-weight: 800; margin: 0 0 8px;
    background: linear-gradient(90deg, #00c8ff 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #4455aa; font-size: .78rem; letter-spacing: .15em; margin: 0; }
.badge {
    display: inline-block; margin-bottom: 14px;
    background: rgba(0,200,255,.08); border: 1px solid rgba(0,200,255,.25);
    color: #00c8ff; border-radius: 99px; padding: 3px 16px;
    font-size: .72rem; letter-spacing: .14em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b0e22 !important;
    border-right: 1px solid rgba(255,255,255,.06) !important;
}
section[data-testid="stSidebar"] * { color: #8899cc !important; }
section[data-testid="stSidebar"] h3 { color: #00c8ff !important; font-size: .85rem !important; letter-spacing: .1em; }

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(0,0,0,.45) !important;
    border: 1px solid rgba(255,255,255,.09) !important;
    border-radius: 10px !important;
    color: #dde0ff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .88rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(0,200,255,.4) !important;
    box-shadow: 0 0 0 2px rgba(0,200,255,.1) !important;
}
label { color: #5566aa !important; font-size: .75rem !important; letter-spacing: .1em; }

/* Primary button */
.stButton > button {
    background: linear-gradient(135deg, #00c8ff, #a78bfa) !important;
    border: none !important; border-radius: 10px !important;
    color: #07091a !important; font-family: 'DM Mono', monospace !important;
    font-weight: 700 !important; font-size: .88rem !important;
    padding: 12px 28px !important; transition: opacity .2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: .82 !important; }

/* Example buttons */
.example-btn > button {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.09) !important;
    color: #7788bb !important; font-size: .78rem !important;
    padding: 6px 14px !important; width: auto !important;
    border-radius: 99px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid rgba(255,255,255,.07); }
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important; color: #445588 !important;
    font-size: .78rem !important; letter-spacing: .08em !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] { color: #00c8ff !important; border-bottom: 2px solid #00c8ff !important; }

/* Progress */
.stProgress > div > div > div { background: linear-gradient(90deg, #00c8ff, #a78bfa) !important; }

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px; padding: 18px; text-align: center;
}
.metric-num {
    font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #00c8ff, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label { color: #445588; font-size: .7rem; letter-spacing: .12em; margin-top: 4px; }

/* Paper card */
.paper-card {
    background: rgba(0,0,0,.3);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px; padding: 16px; margin-bottom: 12px;
}
.paper-title { color: #00c8ff; font-weight: 600; font-size: .88rem; margin-bottom: 4px; }
.paper-meta  { color: #445588; font-size: .78rem; margin-bottom: 6px; }
.paper-abs   { color: #7788aa; font-size: .78rem; line-height: 1.6; }

/* Text area output */
.stTextArea textarea { background: rgba(0,0,0,.35) !important; line-height: 1.8 !important; }

/* Download button */
.stDownloadButton > button {
    background: rgba(52,211,153,.1) !important;
    border: 1px solid rgba(52,211,153,.3) !important;
    color: #34d399 !important; border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important; width: 100%;
}

/* Divider */
hr { border-color: rgba(255,255,255,.06) !important; }

/* Step badges */
.step {
    display: inline-block; width: 24px; height: 24px; line-height: 24px;
    background: rgba(0,200,255,.15); border: 1px solid rgba(0,200,255,.3);
    border-radius: 50%; color: #00c8ff; font-size: .75rem;
    text-align: center; margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# LOAD MODELS  (cached — runs once)
# ══════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    if USE_GPU:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb,
            device_map="auto", low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )

    pipe = hf_pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=800, temperature=0.3,
        do_sample=True, repetition_penalty=1.1,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda" if USE_GPU else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return llm, emb

# ══════════════════════════════════════════════════
# PIPELINE HELPERS
# ══════════════════════════════════════════════════
def search_arxiv(query: str, max_results: int) -> list[dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [
        {
            "title":    r.title,
            "authors":  ", ".join(str(a) for a in r.authors[:3]),
            "abstract": r.summary,
            "url":      r.pdf_url,
            "year":     r.published.year,
        }
        for r in client.results(search)
    ]


def extract_text(paper: dict) -> str:
    try:
        resp = requests.get(paper["url"], timeout=20)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(resp.content)
            path = f.name
        doc  = fitz.open(path)
        text = "".join(doc[i].get_text() for i in range(min(6, len(doc))))
        doc.close()
        if len(text) > 300:
            return text
    except Exception:
        pass
    return (
        f"Title: {paper['title']}\n"
        f"Authors: {paper['authors']}\n"
        f"Year: {paper['year']}\n"
        f"{paper['abstract']}"
    )


def build_retriever(papers: list[dict], emb):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVER
    )
    docs = []
    for p in papers:
        for chunk in splitter.split_text(extract_text(p)):
            docs.append(Document(
                page_content=chunk,
                metadata={"title": p["title"], "authors": p["authors"], "year": p["year"]},
            ))
    vs = Chroma.from_documents(docs, embedding=emb, collection_name="rag_papers")
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
    ), len(docs)


def format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata['title']} ({d.metadata['year']})]\n{d.page_content}"
        for d in docs
    )


# ── LangChain LCEL chains ──────────────────────────
def make_rag_chain(llm):
    prompt = ChatPromptTemplate.from_template(
        "You are an expert AI research analyst.\n"
        "Analyze these research papers about: {query}\n\n"
        "Retrieved content:\n{context}\n\n"
        "Extract:\n"
        "1. Main themes and contributions\n"
        "2. Key findings and results\n"
        "3. Methodology comparisons\n"
        "4. Research gaps\n\n"
        "Be specific and cite paper titles."
    )
    return prompt | llm | StrOutputParser()


def make_report_chain(llm):
    prompt = ChatPromptTemplate.from_template(
        "You are a senior academic report writer.\n"
        "Write a structured research report on: {query}\n\n"
        "Papers reviewed:\n{papers_list}\n\n"
        "Analysis:\n{analysis}\n\n"
        "Use exactly this structure:\n"
        "## Executive Summary\n"
        "## Papers Reviewed\n"
        "## Key Themes & Contributions\n"
        "## Main Findings\n"
        "## Research Gaps\n"
        "## Future Directions\n"
        "## Conclusion"
    )
    return prompt | llm | StrOutputParser()


# ── PDF export ─────────────────────────────────────
def _clean(text: str) -> str:
    text = re.sub(r"\*\*|##|###", "", text)
    text = re.sub(
        r"[^\x00-\x7F]+",
        lambda m: m.group().encode("ascii", "ignore").decode(),
        text,
    )
    return text.strip()


def generate_pdf(report: str, query: str) -> str:
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out = f"/tmp/report_{ts}.pdf"
    doc = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    T  = ParagraphStyle("T",  fontSize=20, fontName="Helvetica-Bold",
                        textColor=colors.HexColor("#1a1a6e"),
                        alignment=TA_CENTER, spaceAfter=6)
    S  = ParagraphStyle("S",  fontSize=11, fontName="Helvetica",
                        textColor=colors.HexColor("#555"),
                        alignment=TA_CENTER, spaceAfter=20)
    H2 = ParagraphStyle("H2", fontSize=13, fontName="Helvetica-Bold",
                        textColor=colors.HexColor("#1a1a6e"),
                        backColor=colors.HexColor("#e8ecff"),
                        spaceBefore=14, spaceAfter=6, leftIndent=4)
    BD = ParagraphStyle("BD", fontSize=10, fontName="Helvetica",
                        textColor=colors.HexColor("#333"), leading=16, spaceAfter=4)
    BL = ParagraphStyle("BL", fontSize=10, fontName="Helvetica",
                        textColor=colors.HexColor("#333"),
                        leading=15, leftIndent=16, spaceAfter=3)
    FT = ParagraphStyle("FT", fontSize=8,  fontName="Helvetica",
                        textColor=colors.grey, alignment=TA_CENTER)

    story = [
        Paragraph("Research Report", T),
        Paragraph(_clean(query), S),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1a6e")),
        Spacer(1, 0.3*cm),
        Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", FT
        ),
        Spacer(1, 0.6*cm),
    ]

    for line in report.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.2*cm))
        elif line.startswith(("## ", "### ")):
            story.append(Paragraph(_clean(line.lstrip("# ")), H2))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph(f"\u2022 {_clean(line[2:])}", BL))
        else:
            story.append(Paragraph(_clean(line), BD))

    doc.build(story)
    return out


# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    max_papers = st.slider("Max Papers", 3, 10, 5)
    st.divider()

    st.markdown("### 💡 Quick Examples")
    examples = [
        "Large Language Models RAG",
        "Transformer self-attention NLP",
        "Medical AI diagnosis imaging",
        "RLHF reinforcement learning",
        "Graph neural networks",
    ]
    for ex in examples:
        with st.container():
            if st.button(ex, key=f"ex_{ex}", use_container_width=False):
                st.session_state["query_val"] = ex

    st.divider()
    st.markdown(
        "<div style='color:#334466;font-size:.72rem;line-height:1.8'>"
        "🤗 Mistral-7B via HuggingFace<br>"
        "🔗 LangChain v0.3+<br>"
        "🗄 ChromaDB + BGE embeddings<br>"
        "📚 arXiv API<br>"
        "📄 ReportLab PDF"
        "</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="badge">FREE · OPEN SOURCE · NO API KEY NEEDED</div>
    <h1>🤖 Research Augmentation Agent</h1>
    <p>AUTONOMOUS MULTI-AGENT SCIENTIFIC LITERATURE ANALYSIS</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════
# QUERY INPUT
# ══════════════════════════════════════════════════
default_q = st.session_state.get("query_val", "")
col_in, col_btn = st.columns([5, 1])

with col_in:
    query = st.text_input(
        "RESEARCH QUERY",
        value=default_q,
        placeholder="e.g., Large Language Models and RAG architectures...",
    )
with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run = st.button("▶ Run", use_container_width=True)

# ══════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════
if run and query.strip():

    with st.spinner("Loading models (first run takes ~5 min)..."):
        llm, emb = load_models()

    rag_chain    = make_rag_chain(llm)
    report_chain = make_report_chain(llm)

    bar    = st.progress(0)
    status = st.empty()

    # 1 ── Search
    status.markdown("**<span class='step'>1</span> Searching arXiv...**",
                    unsafe_allow_html=True)
    papers = search_arxiv(query, max_papers)
    bar.progress(18)

    # 2 ── Process + Embed
    status.markdown("**<span class='step'>2</span> Processing & embedding papers...**",
                    unsafe_allow_html=True)
    retriever, n_chunks = build_retriever(papers, emb)
    bar.progress(42)

    # 3 ── RAG
    status.markdown("**<span class='step'>3</span> RAG Agent analyzing...**",
                    unsafe_allow_html=True)
    docs     = retriever.invoke(query)
    context  = format_docs(docs)
    analysis = rag_chain.invoke({"query": query, "context": context[:3000]})
    bar.progress(68)

    # 4 ── Report
    status.markdown("**<span class='step'>4</span> Writing research report...**",
                    unsafe_allow_html=True)
    papers_list = "\n".join(f"- {p['title']} ({p['year']})" for p in papers)
    report = report_chain.invoke({
        "query": query, "papers_list": papers_list, "analysis": analysis,
    })
    bar.progress(88)

    # 5 ── PDF
    status.markdown("**<span class='step'>5</span> Generating PDF...**",
                    unsafe_allow_html=True)
    pdf_path = generate_pdf(report, query)
    bar.progress(100)
    status.success("✅ Pipeline complete!")

    # ── Metrics ─────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, num, lbl in zip(
        cols,
        [len(papers), n_chunks, len(docs), len(report.split())],
        ["PAPERS", "CHUNKS", "RAG DOCS", "REPORT WORDS"],
    ):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-num">{num}</div>'
            f'<div class="metric-label">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Papers Found",
        "🤖 RAG Analysis",
        "📊 Full Report",
        "📄 Download PDF",
    ])

    with tab1:
        for p in papers:
            st.markdown(
                f'<div class="paper-card">'
                f'<div class="paper-title">[{p["year"]}] {p["title"]}</div>'
                f'<div class="paper-meta">{p["authors"]}</div>'
                f'<div class="paper-abs">{p["abstract"][:220]}...</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with tab2:
        st.text_area("RAG Analysis", value=analysis, height=420, disabled=True)

    with tab3:
        st.markdown(report)

    with tab4:
        st.markdown("### 📄 Your Report is Ready")
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️  Download PDF Report",
                data=f.read(),
                file_name=f"research_report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
            )
        st.markdown(
            "<p style='color:#334466;font-size:.8rem;margin-top:12px'>"
            "PDF includes: title page, styled headings, bullet points, page numbers."
            "</p>",
            unsafe_allow_html=True,
        )
