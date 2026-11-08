import base64
import html
import os
import re
from pathlib import Path

import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ----------------------------
# SETTINGS (change these)
# ----------------------------
CLONE_NAME = "Adeena Azam"
CLONE_DESC = "BS Computer Science Student"
TAGLINE = "Aspiring Software Engineer  ·  AI enthusiast"
INTRO = ("Ask me about my projects, skills, studies and the way I work. "
         "I answer only from notes I wrote myself, and I will say so when I don't know.")
FACTS = ["7th semester", "UMT Sialkot", "Graduating 2027"]
TONE = "friendly, simple and honest"

BASE_DIR = Path(__file__).parent
DATA_FOLDER = BASE_DIR / "data"
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"
ASSETS_FOLDER = BASE_DIR / "assets"   # optional: put avatar.png (your photo) here

CHUNK_SIZE = 250        # words per chunk
OVERLAP = 40            # words shared between neighbouring chunks
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60
TOP_K = 3
MODES = ["BM25 only", "Dense only", "Hybrid (RRF)"]
MODE_HINTS = {
    "BM25 only": "Keyword search: finds my notes that contain the exact words of your question.",
    "Dense only": "Meaning search: finds notes with a similar meaning, even when the words are different.",
    "Hybrid (RRF)": "Both searches combined into one ranking with Reciprocal Rank Fusion.",
}

# llama-3.1-8b-instant was shut down for free users on 16 Aug 2026.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

SUGGESTIONS = [
    "What are your technical skills?",
    "Tell me about SentinelGuard",
    "Tell me about Campus Marketplace",
    "How do you recharge when you feel drained?",
    "What advice do you give to juniors?",
    "What is your career goal?",
]

st.set_page_config(page_title=f"{CLONE_NAME} | AI Clone", page_icon="🤖", layout="centered")


# ----------------------------
# 1. LOAD AND CHUNK DOCUMENTS
# ----------------------------
def create_chunks(text, filename):
    words = text.split()
    chunks, start, chunk_id = [], 0, 1
    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append({
            "filename": filename,
            "chunk_id": chunk_id,
            "text": " ".join(words[start:end]),
        })
        if end >= len(words):
            break
        start = end - OVERLAP
        chunk_id += 1
    return chunks


def tokenize(text):
    # lowercase and keep only letters/numbers, so "skills?" becomes "skills"
    return re.findall(r"[a-z0-9]+", text.lower())


# ----------------------------
# 2. BUILD INDEXES (runs once, then cached)
# ----------------------------
@st.cache_resource(show_spinner="Loading my notes and models...")
def load_everything():
    chunks = []
    files = sorted(list(DATA_FOLDER.glob("*.md")) + list(DATA_FOLDER.glob("*.txt")))
    for path in files:
        chunks.extend(create_chunks(path.read_text(encoding="utf-8"), path.name))
    texts = [c["text"] for c in chunks]

    bm25 = BM25Okapi([tokenize(t) for t in texts]) if chunks else None
    model = SentenceTransformer(EMBED_MODEL_NAME)

    emb = None
    if chunks:
        if EMBEDDINGS_FILE.exists():
            saved = np.load(EMBEDDINGS_FILE)
            if saved.shape[0] == len(texts):
                emb = saved
        if emb is None:   # missing or out of date: build and save
            emb = model.encode(texts, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
            try:
                np.save(EMBEDDINGS_FILE, emb)
            except OSError:
                pass
    return {"chunks": chunks, "n_docs": len(files), "bm25": bm25, "model": model, "emb": emb}


S = load_everything()
all_chunks = S["chunks"]


# ----------------------------
# 3. RETRIEVERS
# ----------------------------
def bm25_rank(query):
    scores = S["bm25"].get_scores(tokenize(query))
    return np.argsort(scores)[::-1], scores


def dense_rank(query):
    q = S["model"].encode(query, convert_to_numpy=True, normalize_embeddings=True)
    scores = S["emb"] @ q          # cosine similarity (vectors are normalized)
    return np.argsort(scores)[::-1], scores


def rrf_fuse(bm25_order, dense_order, k=RRF_K):
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank)."""
    scores = {}
    for order in (bm25_order, dense_order):
        for rank, idx in enumerate(order, start=1):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(query, mode, top_k):
    top_k = int(top_k)
    b_order, b_scores = bm25_rank(query)
    d_order, d_scores = dense_rank(query)
    b_rank = {int(i): r for r, i in enumerate(b_order, 1)}
    d_rank = {int(i): r for r, i in enumerate(d_order, 1)}

    if mode == "BM25 only":
        picked = [(int(i), float(b_scores[i])) for i in b_order[:top_k]]
        label = "BM25 score"
    elif mode == "Dense only":
        picked = [(int(i), float(d_scores[i])) for i in d_order[:top_k]]
        label = "Cosine similarity"
    else:
        picked = rrf_fuse(b_order, d_order)[:top_k]
        label = "RRF score"

    return [{"chunk": all_chunks[idx], "score": score, "label": label,
             "bm25_rank": b_rank[idx], "dense_rank": d_rank[idx]}
            for idx, score in picked]


# ----------------------------
# 4. GENERATION (Groq LLM)
# ----------------------------
PERSONA_PROMPT = """You are the AI clone of {name}, {desc}.
Speak in the first person, in a {tone} tone. Use short sentences.

Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers or other private details.
4. If asked whether you are the real person, say you are an AI clone of {name}.

CONTEXT:
{context}

QUESTION:
{question}"""


def get_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def generate_answer(question, results):
    context = "\n\n".join(
        f"[Source: {r['chunk']['filename']}, chunk {r['chunk']['chunk_id']}]\n{r['chunk']['text']}"
        for r in results
    )
    api_key = get_api_key()
    if not api_key:
        return ("(No GROQ_API_KEY secret found, so this is a templated answer.)\n\n"
                "Most relevant note I found:\n\n" + results[0]["chunk"]["text"][:500])
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = PERSONA_PROMPT.format(name=CLONE_NAME, desc=CLONE_DESC, tone=TONE,
                                       context=context, question=question)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip() or "I don't have that in my notes."
    except Exception as e:
        return (f"Sorry, the language model failed ({type(e).__name__}). "
                f"Here is the top retrieved note instead:\n\n" + results[0]["chunk"]["text"][:500])


# ----------------------------
# 5. LOOK AND FEEL
# ----------------------------
# Palette: ivory #fbf8f3, ink #2a1f2d, plum #4a2545, rose #c0506f, blush #f7e9ec, line #eadfe3
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root, html { color-scheme: only light; }
/* Safety net for dark mode: every plain text element gets dark ink by default.
   :where() keeps this at zero specificity, so the specific colours below still win. */
:where(.stApp) :where(p, span, div, label, li, b, strong, em, i, small, a, summary, td, th, h1, h2, h3, h4) { color: #2a1f2d !important; }
[data-testid="stToolbar"], [data-testid="stStatusWidget"], [data-testid="stDecoration"] { display: none !important; }
html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #fbf8f3 !important; color: #2a1f2d !important; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 880px; padding-top: 1.4rem; padding-bottom: 3rem; }

/* ---------- top bar (AI clone notice) ---------- */
.topbar { background: #4a2545; color: #fbf8f3 !important; border-radius: 12px; padding: 10px 18px;
  font-size: .86rem; font-weight: 500; letter-spacing: .2px; display: flex; align-items: center;
  justify-content: center; gap: 10px; }
.topbar .dot { width: 8px; height: 8px; border-radius: 50%; background: #f0a3b8; display: inline-block; }

/* ---------- hero ---------- */
.hero { display: flex; align-items: center; justify-content: space-between; gap: 36px; padding: 46px 6px 30px 6px; }
.hero-text { flex: 1; }
.eyebrow { text-transform: uppercase; letter-spacing: 3px; font-size: .74rem; color: #a8385a !important; font-weight: 600; }
.name { font-family: 'Playfair Display', serif; font-size: 3.3rem; line-height: 1.05; color: #2a1f2d !important;
  font-weight: 700; margin: 10px 0 8px 0; }
.role { font-size: 1.05rem; color: #4a2545 !important; font-weight: 500; }
.intro { margin-top: 16px; font-size: .98rem; line-height: 1.7; color: #5d5160 !important; max-width: 470px; }
.facts { margin-top: 20px; display: flex; flex-wrap: wrap; gap: 8px; }
.facts span { border: 1px solid #e3d3d9; background: #fff; color: #4a2545 !important; border-radius: 999px;
  padding: 5px 14px; font-size: .8rem; font-weight: 500; }
.hero-art { position: relative; width: 250px; height: 300px; flex: 0 0 250px; }
.hero-art::before { content: ""; position: absolute; inset: 18px -14px -14px 18px; border-radius: 125px 125px 28px 28px;
  background: #f3d9e0; }
.arch { position: relative; width: 100%; height: 100%; border-radius: 125px 125px 28px 28px; overflow: hidden;
  background: #ede3f1; border: 4px solid #fff; box-shadow: 0 18px 40px rgba(74,37,69,.18); }
.arch svg, .arch img, .avatar-sm svg, .avatar-sm img { width: 100%; height: 100%; object-fit: cover; display: block; }

/* ---------- info strip ---------- */
.strip { display: flex; justify-content: center; gap: 0; border-top: 1px solid #eadfe3; border-bottom: 1px solid #eadfe3;
  padding: 14px 0; margin: 4px 0 8px 0; }
.strip span { padding: 0 28px; font-size: .84rem; color: #7a6e7d !important; border-right: 1px solid #eadfe3; }
.strip span:last-child { border-right: none; }
.strip b { color: #4a2545 !important; font-family: 'Playfair Display', serif; font-size: 1.25rem; margin-right: 6px; }

/* ---------- section titles ---------- */
.sec { display: flex; align-items: center; gap: 12px; margin: 38px 0 14px 0; }
.sec .num { font-family: 'Playfair Display', serif; font-size: 1.05rem; color: #a8385a !important; font-weight: 600; }
.sec .ttl { font-family: 'Playfair Display', serif; font-size: 1.35rem; color: #2a1f2d !important; font-weight: 600; }
.sec .rule { flex: 1; height: 1px; background: #eadfe3; }

/* ---------- buttons ---------- */
.stButton, [data-testid="stButton"], [data-testid="stElementContainer"]:has(.stButton) { width: 100% !important; }
.stButton > button { width: 100% !important; }
.stButton > button { border-radius: 12px !important; border: 1px solid #e3d3d9 !important; background: #fff !important;
  color: #4a2545 !important; font-weight: 500; font-size: .88rem; padding: .55rem 1rem; width: 100%;
  min-height: 46px; transition: all .15s ease; }
.stButton > button p, .stButton > button div, .stButton > button span { color: inherit !important; margin: 0; white-space: normal; }
.stButton > button:hover { border-color: #a8385a !important; color: #a8385a !important; background: #fff8fa !important; }
[class*="st-key-sg"] button { justify-content: flex-start; text-align: left; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: #4a2545 !important; border-color: #4a2545 !important; color: #fff !important; }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
  background: #5d3158 !important; color: #fff !important; }
[class*="st-key-act_ask"] button { background: #c0506f !important; border-color: #a8385a !important; color: #fff !important;
  font-weight: 600; font-size: .95rem; min-height: 50px; }
[class*="st-key-act_ask"] button:hover { background: #a93f5d !important; color: #fff !important; }
[class*="st-key-act_cmp"] button { min-height: 50px; }

/* ---------- text input ---------- */
[data-baseweb="input"], [data-baseweb="base-input"] { background: #fff !important; border-radius: 14px !important; }
[data-baseweb="input"] { border: 1.5px solid #d9c6ce !important; }
[data-baseweb="input"]:focus-within { border-color: #a8385a !important; box-shadow: 0 0 0 3px rgba(192,80,111,.15) !important; }
.stTextInput input { color: #2a1f2d !important; -webkit-text-fill-color: #2a1f2d !important; caret-color: #c0506f;
  font-size: 1rem; padding: 14px 16px !important; background: #fff !important; }
.stTextInput input::placeholder { color: #756a79 !important; -webkit-text-fill-color: #756a79 !important; opacity: 1 !important; }

.modelabel { font-size: .78rem; text-transform: uppercase; letter-spacing: 1.6px; color: #7a6e7d !important;
  font-weight: 600; margin: 20px 0 8px 0; }
.modehint { font-size: .86rem; color: #7a6e7d !important; margin: 8px 0 18px 2px; font-style: italic; }

/* ---------- answer ---------- */
.qline { font-size: .9rem; color: #7a6e7d !important; margin-bottom: 12px; }
.qline b { color: #4a2545 !important; font-weight: 600; margin-right: 6px; text-transform: uppercase;
  letter-spacing: 1.4px; font-size: .72rem; }
.answer { background: #fff; border: 1px solid #eadfe3; border-left: 4px solid #c0506f; border-radius: 16px;
  padding: 22px 24px; box-shadow: 0 10px 28px rgba(74,37,69,.07); }
.a-head { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.avatar-sm { width: 44px; height: 44px; flex: 0 0 44px; border-radius: 50%; overflow: hidden; background: #ede3f1;
  border: 2px solid #f3d9e0; }
.a-name { font-family: 'Playfair Display', serif; font-weight: 600; color: #2a1f2d !important; font-size: 1.02rem; line-height: 1.2; }
.a-sub { font-size: .74rem; color: #7a6e7d !important; }
.a-body { font-size: 1rem; line-height: 1.75; color: #2a1f2d !important; }

/* ---------- source cards ---------- */
.card { background: #fff; border: 1px solid #eadfe3; border-radius: 14px; padding: 16px 18px; margin: 12px 0; }
.card-top { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 12px; }
.pill { font-size: .72rem; padding: 3px 11px; border-radius: 999px; background: #f6f1f3; color: #5d5160 !important; }
.pill.src { background: #f3e9f5; color: #4a2545 !important; font-weight: 600; }
.pill.rank { background: #fdecef; color: #a93f5d !important; }
.scoreline { display: flex; justify-content: space-between; font-size: .76rem; color: #7a6e7d !important; margin-bottom: 5px; }
.scoreline b { color: #4a2545 !important; }
.bar { height: 6px; background: #f3e9ec; border-radius: 999px; overflow: hidden; margin-bottom: 12px; }
.bar div { height: 100%; background: #c0506f; border-radius: 999px; }
.snip { font-size: .88rem; color: #4b4150 !important; line-height: 1.65; }
details { font-size: .86rem; color: #4b4150 !important; margin-top: 8px; }
details summary { cursor: pointer; color: #a8385a !important; font-weight: 600; }
details div { margin-top: 8px; line-height: 1.65; color: #4b4150 !important; }

/* ---------- tabs, spinner, notice ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #eadfe3; }
.stTabs [data-baseweb="tab"] { background: transparent !important; }
.stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] div { color: #7a6e7d !important; font-weight: 600; }
.stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] div { color: #a8385a !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #a8385a !important; }
[data-testid="stSpinner"] *, .stSpinner * { color: #4a2545 !important; }
.notice { background: #fff4e5; border: 1px solid #f0d3a5; color: #7a4b0b !important; border-radius: 12px;
  padding: 12px 16px; font-size: .9rem; margin: 14px 0; }
.footer { text-align: center; color: #6f6473 !important; font-size: .78rem; margin: 46px 0 6px 0; line-height: 1.7; }

@media (max-width: 700px) {
  .hero { flex-direction: column-reverse; text-align: center; padding-top: 30px; }
  .hero-art { width: 200px; height: 240px; flex: 0 0 240px; }
  .name { font-size: 2.5rem; }
  .facts { justify-content: center; }
  .intro { margin-left: auto; margin-right: auto; }
  .strip span { padding: 0 12px; }
}
"""

AVATAR_SVG = """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
<rect width="200" height="200" fill="#ede3f1"/>
<path d="M30 200c0-38 30-58 70-58s70 20 70 58z" fill="#4a2545"/>
<path d="M82 142l18 26 18-26z" fill="#ffffff"/>
<rect x="88" y="118" width="24" height="30" rx="10" fill="#e8b894"/>
<path d="M50 98c0-42 21-68 50-68s50 26 50 68v44c0 9-6 15-15 15H65c-9 0-15-6-15-15z" fill="#2b1b17"/>
<ellipse cx="100" cy="95" rx="36" ry="42" fill="#f3cfae"/>
<path d="M60 88c4-32 22-46 41-46 22 0 39 16 39 46-10-15-24-23-40-23-15 0-29 8-40 23z" fill="#2b1b17"/>
<circle cx="86" cy="99" r="3.6" fill="#2b1b17"/>
<circle cx="114" cy="99" r="3.6" fill="#2b1b17"/>
<path d="M90 117q10 9 20 0" stroke="#b4535a" stroke-width="3" fill="none" stroke-linecap="round"/>
<circle cx="77" cy="110" r="6" fill="#f4a3a8" opacity=".45"/>
<circle cx="123" cy="110" r="6" fill="#f4a3a8" opacity=".45"/>
</svg>"""


def avatar_inner():
    """Use assets/avatar.(png/jpg/jpeg/webp) if it exists, otherwise the drawn avatar."""
    for ext, mime in (("png", "png"), ("jpg", "jpeg"), ("jpeg", "jpeg"), ("webp", "webp")):
        p = ASSETS_FOLDER / f"avatar.{ext}"
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<img src="data:image/{mime};base64,{b64}" alt="avatar">'
    return AVATAR_SVG


AVATAR = avatar_inner()


def clean(s):
    """Remove indentation and blank lines so Markdown treats the text as HTML, not code."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def html_block(s):
    st.markdown(clean(s), unsafe_allow_html=True)


def fmt(text):
    t = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return t.replace("\r", "").replace("\n", "<br>")


def section(num, title):
    html_block(f'<div class="sec"><span class="num">{num}</span><span class="ttl">{html.escape(title)}</span><span class="rule"></span></div>')


def render_chunks(results):
    top = max((r["score"] for r in results), default=0) or 1
    for r in results:
        c = r["chunk"]
        if r["label"] == "Cosine similarity":
            width = max(0.0, min(1.0, r["score"])) * 100
        else:
            width = max(0.0, r["score"] / top) * 100
        text = c["text"]
        snippet = text[:230].rsplit(" ", 1)[0] + "..." if len(text) > 230 else text
        html_block(f"""
        <div class="card">
        <div class="card-top">
        <span class="pill src">{html.escape(c['filename'])}</span>
        <span class="pill">chunk {c['chunk_id']}</span>
        <span class="pill rank">BM25 rank #{r['bm25_rank']}</span>
        <span class="pill rank">Dense rank #{r['dense_rank']}</span>
        </div>
        <div class="scoreline"><span>{r['label']}</span><b>{r['score']:.4f}</b></div>
        <div class="bar"><div style="width:{width:.0f}%"></div></div>
        <div class="snip">{fmt(snippet)}</div>
        <details><summary>Read full chunk</summary><div>{fmt(text)}</div></details>
        </div>
        """)


def render_answer(question, answer, mode):
    html_block(f"""
    <div class="qline"><b>You asked</b>{fmt(question)}</div>
    <div class="answer">
    <div class="a-head">
    <div class="avatar-sm">{AVATAR}</div>
    <div><div class="a-name">{html.escape(CLONE_NAME)}</div><div class="a-sub">AI clone, answered using {html.escape(mode)}</div></div>
    </div>
    <div class="a-body">{fmt(answer)}</div>
    </div>
    """)


# ----------------------------
# 6. PAGE
# ----------------------------
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

facts_html = "".join(f"<span>{html.escape(f)}</span>" for f in FACTS)
html_block(f"""
<div class="topbar"><span class="dot"></span>This is an AI clone, not the real person.</div>
<div class="hero">
<div class="hero-text">
<div class="eyebrow">Digital Clone</div>
<div class="name">{html.escape(CLONE_NAME)}</div>
<div class="role">{html.escape(TAGLINE)}</div>
<div class="intro">{html.escape(INTRO)}</div>
<div class="facts">{facts_html}</div>
</div>
<div class="hero-art"><div class="arch">{AVATAR}</div></div>
</div>
<div class="strip">
<span><b>{S['n_docs']}</b>notes</span>
<span><b>{len(all_chunks)}</b>passages</span>
<span><b>3</b>search modes</span>
</div>
""")

if "mode" not in st.session_state:
    st.session_state["mode"] = MODES[2]


def trigger(action):
    st.session_state["action"] = action


def use_suggestion(text):
    st.session_state["q"] = text
    st.session_state["action"] = "ask"


def set_mode(m):
    st.session_state["mode"] = m


section("01", "Start with a question")
cols = st.columns(2)
for i, s in enumerate(SUGGESTIONS):
    cols[i % 2].button(s, key=f"sg{i}", on_click=use_suggestion, args=(s,))

section("02", "Or ask me anything")
st.text_input("Your question", key="q", placeholder="Type your question and press Enter...",
              label_visibility="collapsed", on_change=trigger, args=("ask",))

html_block('<div class="modelabel">Search mode</div>')
mcols = st.columns(3)
for i, m in enumerate(MODES):
    mcols[i].button(m, key=f"mode_{i}", on_click=set_mode, args=(m,),
                    type="primary" if st.session_state["mode"] == m else "secondary")
mode = st.session_state["mode"]
html_block(f'<div class="modehint">{html.escape(MODE_HINTS[mode])}</div>')

b1, b2 = st.columns(2)
b1.button("Ask me", key="act_ask", on_click=trigger, args=("ask",))
b2.button("Compare all 3 modes", key="act_cmp", on_click=trigger, args=("compare",))

# ---- run the action ----
action = st.session_state.pop("action", None)
question = st.session_state.get("q", "").strip()
if action in ("ask", "compare"):
    if not question:
        st.session_state["last"] = None
        html_block('<div class="notice">Please type a question first.</div>')
    elif not all_chunks:
        html_block('<div class="notice">No documents found in the data folder.</div>')
    elif action == "ask":
        results = retrieve(question, mode, TOP_K)
        with st.spinner("Thinking..."):
            answer = generate_answer(question, results)
        st.session_state["last"] = {"type": "ask", "q": question, "answer": answer,
                                    "results": results, "mode": mode}
    else:
        st.session_state["last"] = {"type": "compare", "q": question,
                                    "tabs": {m: retrieve(question, m, TOP_K) for m in MODES}}

last = st.session_state.get("last")
if last:
    if last["type"] == "ask":
        section("03", "My answer")
        render_answer(last["q"], last["answer"], last["mode"])
        section("04", "Where it came from")
        render_chunks(last["results"])
    else:
        section("03", "Comparing the three modes")
        html_block(f'<div class="qline"><b>Question</b>{fmt(last["q"])}</div>')
        tabs = st.tabs(["BM25", "Dense", "Hybrid (RRF)"])
        for tab, m in zip(tabs, MODES):
            with tab:
                render_chunks(last["tabs"][m])

html_block("""
<div class="footer">Built with Python, Streamlit, BM25, Sentence-Transformers, Reciprocal Rank Fusion and Groq.<br>
Every answer comes only from my own notes.</div>
""")