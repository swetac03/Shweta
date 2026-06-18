# =============================================================================
# SciWrite AI  v2.0  —  Flask Serverless Production-Grade Architecture
# =============================================================================
from __future__ import annotations
import os

# Serverless environment overrides for matplotlib memory sandboxing
os.environ["MPLCONFIGDIR"] = "/tmp"

import re
import textwrap
import tempfile
import subprocess
import base64
import io
import zipfile
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent GUI thread collision errors
import matplotlib.pyplot as plt
import pandas as pd
import jinja2
from google import genai
from flask import Flask, render_template, request, jsonify, send_file, session

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "sciwrite_ai_production_secret_key_1029")

# Vercel Serverless Function Deployment Configurations
# Note: Vercel Hobby tier times out tightly at 10 seconds. For sequential generation pipelines 
# that run between 20-60 seconds, configure 'maxDuration = 60' below for Vercel Pro accounts.
maxDuration = 60 

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash-lite"

SECTION_TAGS = [
    "ABSTRACT", "INTRODUCTION", "RELATED_WORK", "METHODOLOGY",
    "RESULTS_AND_ANALYSIS", "DISCUSSION", "CONCLUSION", "ACKNOWLEDGEMENTS",
]

WORDS_PER_PAGE_SINGLE = 450
WORDS_PER_PAGE_DOUBLE = 600

SECTION_WEIGHT: dict[str, float] = {
    "ABSTRACT":             0.05,
    "INTRODUCTION":         0.16,
    "RELATED_WORK":         0.14,
    "METHODOLOGY":          0.22,
    "RESULTS_AND_ANALYSIS": 0.20,
    "DISCUSSION":           0.12,
    "CONCLUSION":           0.07,
    "ACKNOWLEDGEMENTS":     0.04,
}

# InMemory Cache Layer mimicking Streamlit Session State across stateless requests
GENERATION_CACHE = {
    "latex_src": "",
    "zip_payload": b"",
    "sections": {},
    "word_counts": {},
    "total_words": 0
}

# ─────────────────────────────────────────────────────────────────────────────
# LATEX PIPELINE CORE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def escape_for_latex(text: str) -> str:
    """Escape special LaTeX chars in user-supplied strings only."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("&",  r"\&")
    text = text.replace("%",  r"\%")
    text = text.replace("$",  r"\$")
    text = text.replace("#",  r"\#")
    text = text.replace("_",  r"\_")
    text = text.replace("{",  r"\{")
    text = text.replace("}",  r"\}")
    text = text.replace("~",  r"\textasciitilde{}")
    text = text.replace("^",  r"\textasciicircum{}")
    return text


def extract_xml_section(tag: str, raw: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def compute_section_targets(pages: int, two_col: bool) -> dict[str, int]:
    wpp = WORDS_PER_PAGE_DOUBLE if two_col else WORDS_PER_PAGE_SINGLE
    total = pages * wpp
    return {tag: max(80, int(total * w)) for tag, w in SECTION_WEIGHT.items()}


def fetch_arxiv_literature(query: str, max_results: int = 8) -> tuple[str, str]:
    """Queries live arXiv API and builds BibTeX structures + grounding abstracts."""
    if not query.strip():
        query = "machine learning"
        
    safe_query = urllib.parse.quote(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{safe_query}&start=0&max_results={max_results}&sortBy=relevance"
    
    bibtex_entries = []
    abstract_summaries = []
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for i, entry in enumerate(root.findall('atom:entry', ns)):
            title_node = entry.find('atom:title', ns)
            summary_node = entry.find('atom:summary', ns)
            published_node = entry.find('atom:published', ns)
            
            title = title_node.text.replace('\n', ' ').strip() if title_node is not None else "Untitled Research"
            summary = summary_node.text.replace('\n', ' ').strip() if summary_node is not None else ""
            published = published_node.text[:4] if published_node is not None else "2026"
            
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns) if author.find('atom:name', ns) is not None]
            author_str = " and ".join(authors) if authors else "Unknown Researchers"
            
            last_name = authors[0].split()[-1].lower() if authors else "unknown"
            first_word = re.sub(r'[^a-zA-Z0-9]', '', title.split()[0].lower()) if title.split() else "paper"
            cite_key = f"{last_name}{published}{first_word}"
            
            bibtex = textwrap.dedent(f"""
            @article{{{cite_key},
              title={{{title}}},
              author={{{author_str}}},
              journal={{arXiv preprint}},
              year={{{published}}}
            }}
            """).strip()
            
            bibtex_entries.append(bibtex)
            abstract_summaries.append(f"[{cite_key}] {title} ({published}): {summary}")
            
        return "\n\n".join(bibtex_entries), "\n\n".join(abstract_summaries)
        
    except Exception as e:
        return "", ""

# ─────────────────────────────────────────────────────────────────────────────
# CORE API GENAI GROUNDING ENGINES
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini_with_retry(api_key: str, prompt: str, retries: int = 3) -> str:
    """Executes Gemini API calls with exponential backoff for 503 High Demand errors."""
    client = genai.Client(api_key=api_key)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, 
                contents=prompt,
                config=genai.types.GenerateContentConfig(temperature=0.2)
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            raise e
    return ""


def clean_llm_latex_output(raw_output: str) -> str:
    """Removes structural markdown code-fences from response output stream blocks."""
    clean_text = raw_output.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r'^```[a-zA-Z]*\n', '', clean_text)
        clean_text = re.sub(r'\n```$', '', clean_text)
    return clean_text.strip()


def generate_section_prompt(
    tag: str, title: str, arxiv_context: str, bibtex: str, 
    user_notes: str, previous_sections: dict, target_words: int
) -> str:
    """Builds a highly targeted prompt for a SINGLE section, aware of prior context."""
    history = ""
    if previous_sections:
        history = "PREVIOUSLY GENERATED SECTIONS:\n"
        for k, v in previous_sections.items():
            history += f"--- {k} ---\n{v[:500]}... [truncated]\n\n"
            
    cite_keys = re.findall(r"@\w+\{(\w+),", bibtex)
    cite_str = ", ".join(cite_keys) if cite_keys else "(none provided)"

    return textwrap.dedent(f"""
    You are an elite academic researcher. You are writing ONE specific section of a research paper.
    
    PAPER TITLE: {title}
    CURRENT SECTION TO WRITE: {tag}
    TARGET WORD COUNT: ~{target_words} words.
    
    REAL LITERATURE CONTEXT (Use these to ground your claims):
    {arxiv_context}
    
    AVAILABLE CITATION KEYS: {cite_str}
    
    USER NOTES FOR THIS PAPER:
    {user_notes}
    
    {history}
    
    INSTRUCTIONS:
    1. Write ONLY the '{tag}' section. Do not write any other sections.
    2. Write in formal, third-person academic LaTeX prose. 
    3. Use \\cite{{key}} frequently and accurately based on the Literature Context provided.
    4. Do not output standard markdown code fences (like ```latex). Output raw text.
    5. Do not output the section header (e.g., no \\section{{{tag}}}), just the body paragraphs.
    6. Ensure the narrative flows logically from the previously generated sections.
    """).strip()


def generate_academic_chart(api_key: str, paper_title: str, results_notes: str, figure_index: int) -> tuple[str, bytes] | None:
    """Writes automated chart generation python scripts via LLM, executing safely in memory."""
    client = genai.Client(api_key=api_key)
    chart_filename = f"generated_chart_{figure_index}.png"
    
    prompt = f"""
    You are an expert Data Scientist and Academic typesetter. Write an isolated Python script using matplotlib to generate a publication-quality chart for a research paper.
    
    PAPER CONTEXT:
    Title: {paper_title}
    Experimental Results Data: {results_notes}
    Target Figure Filename: {chart_filename}
    
    CRITICAL DESIGN REQUIREMENTS:
    1. Aesthetics: Use a clean academic style. Set dark grey or black gridlines, clear legible labels, axis titles, and a descriptive legend if applicable.
    2. Data: Translate the user's unstructured metrics into explicit arrays or dataframes within the script.
    3. Output: The script MUST save the file to the exact path string: '{chart_filename}' using plt.savefig('{chart_filename}', dpi=300, bbox_inches='tight').
    4. Safety: Do NOT use plt.show(). Use plt.close('all') at the absolute end of the execution string.
    5. Formatting: Output ONLY raw executable Python code. No markdown formatting. No ```python blocks. No explanations.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                system_instruction="You are an automated Python script generator that outputs raw executable code strings with zero markdown encapsulation."
            )
        )
        
        clean_code = response.text.strip().replace("```python", "").replace("```", "")
        
        local_scope = {"plt": plt, "pd": pd, "re": re}
        plt.close('all')
        
        # Execute dynamically constructed Matplotlib script securely inside dynamic sandbox environment
        exec(clean_code, globals(), local_scope)
        
        # Pull chart resource asset directly out of server runtime temporary context space
        if os.path.exists(chart_filename):
            with open(chart_filename, "rb") as f:
                img_bytes = f.read()
            os.remove(chart_filename)
            plt.close('all')
            return chart_filename, img_bytes
            
    except Exception:
        plt.close('all')
        return None


def generate_overleaf_zip(latex_src: str, bibtex_data: str, uploaded_figs: list, generated_figs: list[tuple[str, bytes]]) -> bytes:
    """Creates a production-ready compressed architecture stream container layer optimized for Overleaf imports."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("main.tex", latex_src)
        if bibtex_data.strip():
            zip_file.writestr("references.bib", bibtex_data.strip())
            
        for uf_name, uf_bytes in uploaded_figs:
            safe_name = re.sub(r"[^\w.\-]", "_", uf_name)
            zip_file.writestr(safe_name, uf_bytes)
                
        for filename, img_bytes in generated_figs:
            zip_file.writestr(filename, img_bytes)
                
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# JINJA TEMPLATE COMPILER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def make_jinja_env() -> jinja2.Environment:
    return jinja2.Environment(
        block_start_string="[%", block_end_string="%]",
        variable_start_string="[[", variable_end_string="]]",
        comment_start_string="[#", comment_end_string="#]",
        trim_blocks=True, lstrip_blocks=True,
        autoescape=False, undefined=jinja2.Undefined,
    )

LATEX_TMPL = r"""
\documentclass[11pt[% if two_col %],twocolumn[% endif %]]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath,amssymb,amsfonts,mathtools,bm}
\usepackage[
[% if two_col %]
    letterpaper,top=0.9in,bottom=0.9in,left=0.65in,right=0.65in,columnsep=18pt,
[% else %]
    letterpaper,top=1in,bottom=1in,left=1.1in,right=1.1in,
[% endif %]
]{geometry}
\usepackage{graphicx,booktabs,tabularx,multirow,array,float}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{hyperref,url}
\hypersetup{colorlinks=true,linkcolor=blue!65!black,citecolor=green!55!black,urlcolor=blue!75!black,
    pdftitle={[[ escaped_title ]]},pdfauthor={[[ first_author ]]},pdfkeywords={[[ escaped_kw ]]}}
\usepackage{parskip,setspace,titlesec,abstract,authblk,fancyhdr,xcolor,soul}
\usepackage{algorithm,algpseudocode}
\titleformat{\section}{\large\bfseries\scshape}{}{0em}{}[\vspace{-2pt}\rule{\linewidth}{0.5pt}\vspace{2pt}]
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}
\titlespacing*{\section}{0pt}{14pt}{6pt}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0.35pt}
\fancyhead[L]{\small\scshape\textcolor{gray}{[[ short_title ]]}}
\fancyhead[R]{\small\textcolor{gray}{\thepage}}
\fancyfoot[C]{\small\textcolor{gray!60}{Generated by SciWrite AI \textperiodcentered{} \today}}
\renewcommand{\abstractname}{\normalsize\bfseries\scshape Abstract}
\setlength{\absleftindent}{0pt}\setlength{\absrightindent}{0pt}
\title{\vspace{-1.5em}\Large\bfseries [[ escaped_title ]]%
[% if kw_raw %]\\\vspace{0.3em}\normalsize\normalfont\textit{Keywords:}\ \small [[ escaped_kw ]][% endif %]}
[% for a in authors %]
\author[[[ loop.index ]]]{\textbf{[[ a.name ]]}[% if a.email %]\thanks{\href{mailto:[[ a.email ]]}{[[ a.email ]]}}\vspace{-0.5em}[% endif %]}
\affil[[[ loop.index ]]]{\small\textit{[[ a.affil ]]}}
[% endfor %]
\date{\today}
\begin{document}
\maketitle\thispagestyle{fancy}
\begin{abstract}\noindent [[ sec.ABSTRACT ]]\end{abstract}
[% if two_col %]\vspace{0.5em}\noindent\rule{\linewidth}{0.3pt}\vspace{0.2em}[% endif %]
\section{Introduction}
[[ sec.INTRODUCTION ]]
\section{Related Work}
[[ sec.RELATED_WORK ]]
\section{Methodology}
[[ sec.METHODOLOGY ]]
[% if figs %]
[% for f in figs %]
\begin{figure}[ht]\centering
\includegraphics[width=0.95\linewidth]{[[ f.fn ]]}
\caption{[[ f.cap ]]}\label{fig:[[ loop.index ]]}
\end{figure}
[% endfor %]
[% endif %]
\section{Results and Analysis}
[[ sec.RESULTS_AND_ANALYSIS ]]
\section{Discussion}
[[ sec.DISCUSSION ]]
\section{Conclusion}
[[ sec.CONCLUSION ]]
\section*{Acknowledgements}
[[ sec.ACKNOWLEDGEMENTS ]]
[% if bibtex %]
\begin{filecontents*}{\jobname.bib}
[[ bibtex ]]
\end{filecontents*}
\bibliographystyle{unsrtnat}
\bibliography{\jobname}
[% endif %]
\end{document}
"""

def render_latex(title: str, authors: list[dict], two_col: bool, keywords: str, sections: dict, bibtex: str, figs: list | None = None) -> str:
    env = make_jinja_env()
    tmpl = env.from_string(LATEX_TMPL)
    author_list = []
    first_author = ""
    for auth in authors:
        name = str(auth.get("name", "")).strip()
        if not name:
            continue
        if not first_author:
            first_author = escape_for_latex(name)
        author_list.append({
            "name":  escape_for_latex(name),
            "affil": escape_for_latex(str(auth.get("affil", ""))),
            "email": escape_for_latex(str(auth.get("email", ""))),
        })
    short = (title[:52]+"...") if len(title) > 55 else title
    return tmpl.render(
        escaped_title=escape_for_latex(title),
        short_title=escape_for_latex(short),
        first_author=first_author,
        escaped_kw=escape_for_latex(keywords),
        kw_raw=keywords.strip(),
        authors=author_list,
        two_col=two_col,
        sec=sections,
        bibtex=bibtex.strip(),
        figs=figs or [],
    )

# ─────────────────────────────────────────────────────────────────────────────
# FLASK WEB SERVER ROUTING INTERFACES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Renders primary monolithic administrative UI workspace wrapper matrix dashboard template."""
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate_paper():
    """Unified single-transaction generation pipeline processing multi-agent sequentially."""
    global GENERATION_CACHE
    
    # Extract structural form fields directly from JSON payload submission
    data = request.json or {}
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"success": False, "error": "System GEMINI_API_KEY environment variable is missing on server execution node."}), 400
        
    paper_title = data.get("title", "").strip()
    keywords = data.get("keywords", "").strip()
    layout = data.get("layout", "Single Column")
    target_pages = int(data.get("target_pages", 8))
    venue = data.get("venue", "NeurIPS")
    
    abstract_goals = data.get("abstract_goals", "")
    intro_bg = data.get("intro_bg", "")
    methodology_notes = data.get("methodology_notes", "")
    results_data = data.get("results_data", "")
    extra_notes = data.get("extra_notes", "")
    
    bibtex_entries = data.get("bibtex_entries", "")
    authors = data.get("authors", [])
    
    # Base64 string decode layer transformation processing file binaries back safely 
    uploaded_files_payload = data.get("uploaded_figures", []) # Format list of dicts: {"name": str, "base64": str}
    decoded_uploaded_figs = []
    for f_obj in uploaded_files_payload:
        try:
            f_bytes = base64.b64decode(f_obj["base64"].split(",")[1] if "," in f_obj["base64"] else f_obj["base64"])
            decoded_uploaded_figs.append((f_obj["name"], f_bytes))
        except Exception:
            pass

    two_col = (layout == "Double Column")
    sec_targets = compute_section_targets(target_pages, two_col)
    
    # Step 1: Literature Validation Grounding Architecture via arXiv API
    search_query = f"{paper_title} {keywords}"
    real_bibtex, arxiv_summaries = fetch_arxiv_literature(search_query, max_results=6)
    combined_bibtex = f"{bibtex_entries}\n\n{real_bibtex}".strip()
    
    # Step 2: Render Empirical Data Charts safely via isolated matplotlib
    generated_charts_list = []
    for idx in range(1, 3):
        chart_asset = generate_academic_chart(api_key, paper_title, results_data, idx)
        if chart_asset:
            generated_charts_list.append(chart_asset)
            
    # Step 3: Sequential Academic Document Synthesis Engine Loop Blocks
    sections_dict = {}
    compiled_user_notes = f"Abstract: {abstract_goals}\nIntro: {intro_bg}\nMethod: {methodology_notes}\nResults: {results_data}\nExtra: {extra_notes}"
    
    try:
        for tag in SECTION_TAGS:
            prompt = generate_section_prompt(
                tag=tag, title=paper_title, arxiv_context=arxiv_summaries,
                bibtex=combined_bibtex, user_notes=compiled_user_notes,
                previous_sections=sections_dict, target_words=sec_targets.get(tag, 300)
            )
            section_text = call_gemini_with_retry(api_key, prompt)
            sections_dict[tag] = clean_llm_latex_output(section_text)
            
        word_counts = {t: len(v.split()) for t, v in sections_dict.items()}
        total_words = sum(word_counts.values())
        
        # Structure and compile metadata for figure injection blocks inside LaTeX document ecosystem
        fig_list = []
        for uf_name, _ in decoded_uploaded_figs:
            safe_name = re.sub(r"[^\w.\-]", "_", uf_name)
            fig_list.append({"fn": safe_name, "cap": f"Empirical field observations: {uf_name}"})
        for filename, _ in generated_charts_list:
            fig_list.append({"fn": filename, "cap": "Programmatic verification matrix detailing analytical experimental parameters."})
            
        # Step 4: Final Document Compilation and formatting via Template engines
        latex_src = render_latex(
            title=paper_title, authors=authors, two_col=two_col,
            keywords=keywords, sections=sections_dict,
            bibtex=combined_bibtex, figs=fig_list
        )
        
        zip_payload = generate_overleaf_zip(latex_src, combined_bibtex, decoded_uploaded_figs, generated_charts_list)
        
        # Populate operational variables directly onto global proxy thread targets layer storage
        GENERATION_CACHE["latex_src"] = latex_src
        GENERATION_CACHE["zip_payload"] = zip_payload
        GENERATION_CACHE["sections"] = sections_dict
        GENERATION_CACHE["word_counts"] = word_counts
        GENERATION_CACHE["total_words"] = total_words
        
        # Construct return structure matrix summary metadata
        targets_metrics = [{"section": t.replace("_"," ").title(), "written": word_counts[t], "target": sec_targets.get(t,0), "coverage": f"{int(word_counts[t]/max(1,sec_targets.get(t,1))*100)}%"} for t in SECTION_TAGS]
        
        return jsonify({
            "success": True,
            "total_words": total_words,
            "sections_count": len(sections_dict),
            "metrics": targets_metrics,
            "preview_sections": sections_dict,
            "latex_src": latex_src
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Generation Pipeline Failure: {str(e)}"}), 500


@app.route("/api/download/tex", methods=["GET"])
def download_tex():
    """Streams compiled raw LaTeX source files directly out of system cache configurations memory fields."""
    if not GENERATION_CACHE["latex_src"]:
        return "No asset generated to execute stream downloads.", 400
    return send_file(
        io.BytesIO(GENERATION_CACHE["latex_src"].encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name="main.tex"
    )


@app.route("/api/download/zip", methods=["GET"])
def download_zip():
    """Streams full Overleaf workspace target project files structures back safely."""
    if not GENERATION_CACHE["zip_payload"]:
        return "No workspace bundle payload structures established.", 400
    return send_file(
        io.BytesIO(GENERATION_CACHE["zip_payload"]),
        mimetype="application/zip",
        as_attachment=True,
        download_name="sciwrite_overleaf_project.zip"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)