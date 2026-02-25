"""
Autonomous Research Augmentation Agent
Free Stack: HuggingFace + LangChain + ChromaDB + Gradio
Deploy on: Hugging Face Spaces (FREE)
"""

import os
import re
import arxiv
import fitz
import requests
import chromadb
import datetime
import tempfile
import warnings
warnings.filterwarnings("ignore")

import torch
import gradio as gr

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER

# ══════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════
MODEL_NAME    = "mistralai/Mistral-7B-Instruct-v0.2"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
MAX_PAPERS    = 5
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
TOP_K         = 6
USE_GPU       = torch.cuda.is_available()

# ══════════════════════════════════════════
# LOAD MODELS (once at startup)
# ══════════════════════════════════════════
print("📥 Loading models...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

if USE_GPU:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb,
        device_map="auto", low_cpu_mem_usage=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

hf_pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer,
    max_new_tokens=800, temperature=0.3,
    do_sample=True, repetition_penalty=1.1,
    return_full_text=False
)
llm = HuggingFacePipeline(pipeline=hf_pipe)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda" if USE_GPU else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("✅ Models loaded!")

# ══════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════
parser = StrOutputParser()

rag_prompt = ChatPromptTemplate.from_template("""You are an expert AI research analyst.
Analyze these research papers about: {query}

Retrieved content:
{context}

Extract:
1. Main themes and contributions
2. Key findings and results
3. Methodology comparisons
4. Research gaps

Be specific and cite paper titles.""")

report_prompt = ChatPromptTemplate.from_template("""You are a senior academic report writer.
Write a structured research report on: {query}

Papers reviewed:
{papers_list}

Analysis:
{analysis}

Write the full report using this structure:
## Executive Summary
## Papers Reviewed
## Key Themes & Contributions
## Main Findings
## Research Gaps
## Future Directions
## Conclusion""")

rag_chain    = rag_prompt    | llm | parser
report_chain = report_prompt | llm | parser


def search_arxiv(query, max_results=MAX_PAPERS):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for r in client.results(search):
        papers.append({
            "title":    r.title,
            "authors":  ", ".join(str(a) for a in r.authors[:3]),
            "abstract": r.summary,
            "url":      r.pdf_url,
            "year":     r.published.year,
        })
    return papers


def extract_text(paper):
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
    except:
        pass
    return f"Title: {paper['title']}\nAuthors: {paper['authors']}\nYear: {paper['year']}\n{paper['abstract']}"


def build_vectorstore(papers):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = []
    for p in papers:
        chunks = splitter.split_text(extract_text(p))
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={"title": p["title"], "authors": p["authors"], "year": p["year"]}
            ))
    vs = Chroma.from_documents(docs, embedding=embeddings,
                               collection_name="research_papers")
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3}
    )


def clean_text(text):
    text = text.replace("**", "").replace("##", "").replace("###", "")
    text = re.sub(r'[^\x00-\x7F]+', lambda m: m.group().encode('ascii','ignore').decode(), text)
    return text.strip()


def generate_pdf(report_text, query):
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"/tmp/research_report_{ts}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("Title", fontSize=20, fontName="Helvetica-Bold",
                                 textColor=colors.HexColor("#1a1a6e"),
                                 alignment=TA_CENTER, spaceAfter=6)
    sub_style   = ParagraphStyle("Sub", fontSize=11, fontName="Helvetica",
                                 textColor=colors.HexColor("#555555"),
                                 alignment=TA_CENTER, spaceAfter=20)
    h2_style    = ParagraphStyle("H2", fontSize=13, fontName="Helvetica-Bold",
                                 textColor=colors.HexColor("#1a1a6e"),
                                 backColor=colors.HexColor("#e8ecff"),
                                 spaceBefore=14, spaceAfter=6, leftIndent=4)
    body_style  = ParagraphStyle("Body", fontSize=10, fontName="Helvetica",
                                 textColor=colors.HexColor("#333333"),
                                 leading=16, spaceAfter=4)
    bullet_style = ParagraphStyle("Bullet", fontSize=10, fontName="Helvetica",
                                  textColor=colors.HexColor("#333333"),
                                  leading=15, leftIndent=16, spaceAfter=3)
    footer_style = ParagraphStyle("Footer", fontSize=8, fontName="Helvetica",
                                  textColor=colors.grey, alignment=TA_CENTER)

    story = [
        Paragraph("Research Report", title_style),
        Paragraph(clean_text(query), sub_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1a6e")),
        Spacer(1, 0.3*cm),
        Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", footer_style),
        Spacer(1, 0.6*cm),
    ]

    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.2*cm))
        elif line.startswith("## "):
            story.append(Paragraph(clean_text(line[3:]), h2_style))
        elif line.startswith("### "):
            story.append(Paragraph(clean_text(line[4:]), h2_style))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph(f"• {clean_text(line[2:])}", bullet_style))
        else:
            story.append(Paragraph(clean_text(line), body_style))

    doc.build(story)
    return filename


# ══════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════
def run_research(query, progress=gr.Progress()):
    if not query.strip():
        return "", "", "", None

    # 1. Search
    progress(0.1, desc="🔍 Searching arXiv...")
    papers = search_arxiv(query)
    papers_md = "\n".join([
        f"**[{p['year']}]** {p['title']} — *{p['authors']}*"
        for p in papers
    ])

    # 2. Process + Embed
    progress(0.3, desc="📄 Processing & embedding papers...")
    retriever = build_vectorstore(papers)

    # 3. RAG
    progress(0.55, desc="🤖 RAG Agent analyzing...")
    docs    = retriever.invoke(query)
    context = "\n\n---\n\n".join([
        f"[{d.metadata['title']} ({d.metadata['year']})]\n{d.page_content}"
        for d in docs
    ])
    analysis = rag_chain.invoke({"query": query, "context": context[:3000]})

    # 4. Report
    progress(0.8, desc="📝 Writing report...")
    papers_list = "\n".join([f"- {p['title']} ({p['year']})" for p in papers])
    report = report_chain.invoke({
        "query": query, "papers_list": papers_list, "analysis": analysis
    })

    # 5. PDF
    progress(0.95, desc="📄 Generating PDF...")
    pdf_path = generate_pdf(report, query)

    progress(1.0, desc="✅ Done!")
    return papers_md, analysis, report, pdf_path


# ══════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

body, .gradio-container {
    background: #07091a !important;
    font-family: 'DM Mono', monospace !important;
}

.header-box {
    background: linear-gradient(135deg, #0d1130 0%, #0a0f28 100%);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
}

.header-box h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #00c8ff, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px !important;
}

.header-box p {
    color: #5566aa !important;
    font-size: 0.85rem !important;
    margin: 0 !important;
    letter-spacing: 0.1em;
}

.badge {
    display: inline-block;
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.3);
    color: #00c8ff !important;
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    margin-bottom: 16px;
}

label { color: #7788cc !important; font-size: 0.8rem !important; letter-spacing: 0.08em; }

textarea, input {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #dde0ff !important;
    font-family: 'DM Mono', monospace !important;
}

textarea:focus, input:focus {
    border-color: rgba(0,200,255,0.4) !important;
    box-shadow: 0 0 0 2px rgba(0,200,255,0.1) !important;
}

button.primary {
    background: linear-gradient(135deg, #00c8ff, #a78bfa) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #07091a !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}

button.primary:hover { opacity: 0.85 !important; }

.tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #5566aa !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    padding: 10px 20px !important;
}

.tab-nav button.selected {
    color: #00c8ff !important;
    border-bottom-color: #00c8ff !important;
}

.output-box {
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    color: #c8ccee !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.8 !important;
}

.download-btn {
    background: rgba(52,211,153,0.12) !important;
    border: 1px solid rgba(52,211,153,0.3) !important;
    border-radius: 10px !important;
    color: #34d399 !important;
}
"""

EXAMPLES = [
    ["Large Language Models and Retrieval Augmented Generation"],
    ["Transformer architecture self-attention mechanisms"],
    ["Medical AI diagnosis deep learning"],
    ["Reinforcement learning from human feedback RLHF"],
]

with gr.Blocks(css=CSS, title="Research Agent") as demo:

    gr.HTML("""
    <div class="header-box">
        <div class="badge">FREE · HuggingFace + LangChain + ChromaDB</div>
        <h1>Research Augmentation Agent</h1>
        <p>AUTONOMOUS MULTI-AGENT SCIENTIFIC LITERATURE ANALYSIS</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                placeholder="e.g., Large Language Models and RAG architectures...",
                label="RESEARCH QUERY",
                lines=2,
                elem_classes="output-box"
            )
        with gr.Column(scale=1, min_width=140):
            run_btn = gr.Button("▶ Run Pipeline", variant="primary")

    gr.Examples(examples=EXAMPLES, inputs=query_input, label="Quick Examples")

    with gr.Tabs():
        with gr.Tab("📚 Papers Found"):
            papers_out = gr.Markdown(elem_classes="output-box")

        with gr.Tab("🤖 RAG Analysis"):
            analysis_out = gr.Textbox(
                label="", lines=18, interactive=False,
                elem_classes="output-box"
            )

        with gr.Tab("📊 Full Report"):
            report_out = gr.Markdown(elem_classes="output-box")

        with gr.Tab("📄 Download PDF"):
            pdf_out = gr.File(
                label="Research Report PDF",
                elem_classes="download-btn"
            )

    run_btn.click(
        fn=run_research,
        inputs=query_input,
        outputs=[papers_out, analysis_out, report_out, pdf_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)
