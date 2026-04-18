# =============================================================================
# capstone_streamlit.py — HR Policy Bot | Streamlit UI
# Agentic AI Capstone | Dr. Kanthi Kiran Sirra | 2026
# =============================================================================

import os
import uuid
import streamlit as st
from agent import build_agent

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HRBot — HR Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: #f5f4f0;
        color: #1a1a2e;
    }

    .main-header {
        font-family: 'DM Mono', monospace;
        font-size: 2rem;
        font-weight: 500;
        color: #1a1a2e;
        letter-spacing: -0.5px;
        padding: 1.2rem 0 0.2rem 0;
        border-bottom: 3px solid #1a1a2e;
        margin-bottom: 0.3rem;
    }

    .sub-header {
        color: #666;
        font-size: 0.88rem;
        margin-bottom: 1.5rem;
        font-weight: 300;
        letter-spacing: 0.03em;
    }

    .chat-user {
        background: #1a1a2e;
        color: #f5f4f0;
        padding: 12px 16px;
        border-radius: 2px 16px 16px 16px;
        margin: 10px 0 4px 0;
        font-size: 0.93rem;
        line-height: 1.5;
    }

    .chat-bot {
        background: #ffffff;
        color: #1a1a2e;
        padding: 14px 18px;
        border-radius: 16px 16px 16px 2px;
        margin: 4px 0 10px 0;
        font-size: 0.93rem;
        line-height: 1.6;
        border: 1px solid #e0ddd8;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .meta-row {
        display: flex;
        gap: 6px;
        margin-top: 8px;
        flex-wrap: wrap;
    }

    .pill {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 3px;
        border: 1px solid;
    }

    .pill-route {
        background: #e8f5e9;
        border-color: #4caf50;
        color: #2e7d32;
    }

    .pill-faith {
        background: #fff3e0;
        border-color: #ff9800;
        color: #e65100;
    }

    .pill-source {
        background: #e3f2fd;
        border-color: #2196f3;
        color: #0d47a1;
    }

    .policy-tag {
        display: inline-block;
        background: #fff;
        border: 1px solid #d0ccc5;
        color: #444;
        padding: 3px 9px;
        border-radius: 3px;
        font-size: 0.76rem;
        margin: 2px 2px 2px 0;
        font-family: 'DM Mono', monospace;
    }

    .sidebar-card {
        background: #ffffff;
        border: 1px solid #e0ddd8;
        border-radius: 4px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }

    .sidebar-card h4 {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888;
        margin: 0 0 8px 0;
    }

    .stButton > button {
        background: #1a1a2e;
        color: #f5f4f0;
        border: none;
        border-radius: 3px;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        width: 100%;
        padding: 8px;
        letter-spacing: 0.03em;
        transition: opacity 0.2s;
    }

    .stButton > button:hover { opacity: 0.8; }

    .suggestion-btn button {
        background: #fff !important;
        color: #1a1a2e !important;
        border: 1px solid #d0ccc5 !important;
        border-radius: 3px;
        font-size: 0.8rem !important;
        text-align: left;
        padding: 8px 10px;
    }

    .suggestion-btn button:hover {
        border-color: #1a1a2e !important;
        opacity: 1 !important;
    }

    div[data-testid="stChatInput"] textarea {
        border-radius: 3px;
        border: 1.5px solid #d0ccc5;
        font-family: 'DM Sans', sans-serif;
    }

    code { font-family: 'DM Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED AGENT — built once, reused on every rerun
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    return build_agent(groq_key)

agent_app = get_agent()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "messages"  not in st.session_state: st.session_state.messages  = []
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "last_meta" not in st.session_state: st.session_state.last_meta = {}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏢 HRBot")
    st.markdown("<p style='color:#888;font-size:0.78rem;'>Agentic AI Capstone · 2026</p>",
                unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card"><h4>Policy Coverage</h4>', unsafe_allow_html=True)
    policies = [
        "Annual Leave", "Sick Leave", "WFH Policy",
        "Payroll & Salary", "Maternity / Paternity",
        "Appraisal & PIP", "Code of Conduct",
        "PF & Gratuity", "Resignation & FnF",
        "Training & L&D",
    ]
    for p in policies:
        st.markdown(f"<span class='policy-tag'>📄 {p}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card"><h4>Agent Features</h4>', unsafe_allow_html=True)
    st.markdown("""
- Remembers your name & employee ID  
- RAG from 10 curated policy docs  
- Self-reflection faithfulness check  
- datetime tool for salary/leave context  
- Refuses to fabricate policy details  
- Redirects to HR when unsure  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.last_meta:
        m = st.session_state.last_meta
        st.markdown('<div class="sidebar-card"><h4>Last Response</h4>', unsafe_allow_html=True)
        st.markdown(f"Route: `{m.get('route','—')}`")
        st.markdown(f"Faithfulness: `{m.get('faithfulness',0):.2f}`")
        srcs = m.get("sources", [])
        if srcs:
            st.markdown(f"Source: `{srcs[0]}`")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 New Conversation"):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.last_meta = {}
        st.rerun()

    st.markdown(
        f"<p style='color:#aaa;font-size:0.7rem;text-align:center;margin-top:10px;'>"
        f"Session: {st.session_state.thread_id[:8]}...</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏢 HRBot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your 24/7 HR Policy Assistant · Ask about leave, payroll, benefits, and more</div>',
            unsafe_allow_html=True)

# Render history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-user">👤 <b>You:</b> {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        meta = msg.get("meta", {})
        meta_html = ""
        if meta:
            meta_html = '<div class="meta-row">'
            meta_html += f'<span class="pill pill-route">→ {meta.get("route","—")}</span>'
            meta_html += f'<span class="pill pill-faith">faith: {meta.get("faithfulness",0):.2f}</span>'
            for src in meta.get("sources", [])[:2]:
                meta_html += f'<span class="pill pill-source">📎 {src}</span>'
            meta_html += "</div>"
        st.markdown(
            f'<div class="chat-bot">🤖 <b>HRBot:</b><br><br>{msg["content"]}{meta_html}</div>',
            unsafe_allow_html=True,
        )

# Suggestions when empty
if not st.session_state.messages:
    st.markdown("#### Common questions")
    suggestions = [
        "How many annual leave days do I get?",
        "When is my salary credited each month?",
        "How does the gratuity formula work?",
        "What are the WFH eligibility rules?",
        "What is the notice period for a senior associate?",
        "How many weeks of maternity leave do I get?",
    ]
    cols = st.columns(3)
    for i, sug in enumerate(suggestions):
        with cols[i % 3]:
            st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
            if st.button(sug, key=f"sug_{i}"):
                st.session_state._pending = sug
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def run_question(q: str):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = agent_app.invoke(
        {"question": q, "messages": [], "eval_retries": 0,
         "user_name": "", "employee_id": ""},
        config=config,
    )
    answer = result.get("answer", "Sorry, I could not generate an answer.")
    meta   = {
        "route":       result.get("route", "—"),
        "faithfulness": result.get("faithfulness", 0.0),
        "sources":     result.get("sources", []),
    }
    st.session_state.messages.append({"role": "user",      "content": q})
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
    st.session_state.last_meta = meta
    st.rerun()


# Handle suggestion click
if hasattr(st.session_state, "_pending"):
    pending = st.session_state._pending
    del st.session_state._pending
    with st.spinner("Looking up policy..."):
        run_question(pending)

# Handle chat input
if question := st.chat_input("Ask an HR policy question..."):
    with st.spinner("Checking policy documents..."):
        run_question(question)
