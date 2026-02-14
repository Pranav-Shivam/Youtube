"""
YouTube Transcript Studio - Modern UI with Theme Toggle
A polished, production-ready interface with perfect dark/light mode support.
"""

import math
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import streamlit as st

# Import from the improved backend
from youtube_transcriber_improved import (
    YouTubeTranscriber,
    TranscriberConfig,
    DependencyChecker,
    TranscriptFormat,
    TranscriptWriter,
    TranscriptionMethod,
    TranscriptResult,
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="YouTube Transcript Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "transcript_result" not in st.session_state:
        st.session_state.transcript_result = None
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "last_url" not in st.session_state:
        st.session_state.last_url = ""
    
    if "config" not in st.session_state:
        st.session_state.config = TranscriberConfig(
            enable_cache=True,
            whisper_model="base",
        )
    
    # Theme state - defaults to light
    if "theme" not in st.session_state:
        st.session_state.theme = "light"


# ============================================================================
# THEME STYLES
# ============================================================================

def get_theme_css(theme: str) -> str:
    """Generate CSS based on current theme."""
    
    if theme == "dark":
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Dark Theme Variables */
        :root {
            --primary: #818cf8;
            --primary-dark: #6366f1;
            --primary-light: #a5b4fc;
            --secondary: #a78bfa;
            --success: #34d399;
            --warning: #fbbf24;
            --error: #f87171;
            --bg-main: #0f172a;
            --bg-secondary: #1e293b;
            --surface: #1e293b;
            --surface-elevated: #334155;
            --surface-alt: #0f172a;
            --border: #334155;
            --border-light: #475569;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.5);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.6);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.7);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.8);
        }

        /* Main Background */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-primary);
        }

        /* Hero Section */
        .hero {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            border-radius: 1rem;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-xl);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .hero::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 100%);
            pointer-events: none;
        }

        .hero-content {
            position: relative;
            z-index: 1;
        }

        .hero h1 {
            color: white;
            font-size: 2.25rem;
            font-weight: 700;
            margin: 0 0 0.5rem 0;
            text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        .hero p {
            color: rgba(255, 255, 255, 0.95);
            font-size: 1rem;
            margin: 0;
            line-height: 1.5;
        }

        /* Cards */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-md);
            transition: all 0.2s ease;
        }

        .card:hover {
            box-shadow: var(--shadow-lg);
            border-color: var(--border-light);
        }

        /* Info Cards */
        .info-card {
            background: var(--surface-elevated);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--shadow-sm);
        }

        .info-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.375rem;
        }

        .info-value {
            font-size: 0.9375rem;
            color: var(--text-primary);
            font-weight: 500;
            line-height: 1.4;
        }

        /* Metric Cards */
        .metric-item {
            background: var(--surface-elevated);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.25rem;
            text-align: center;
            box-shadow: var(--shadow-sm);
        }

        .metric-label {
            font-size: 0.8125rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
            line-height: 1;
        }

        .metric-suffix {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        /* Status Badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.375rem;
            padding: 0.375rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8125rem;
            font-weight: 600;
            line-height: 1;
        }

        .status-badge.success {
            background: #064e3b;
            color: #6ee7b7;
            border: 1px solid #065f46;
        }

        .status-badge.warning {
            background: #78350f;
            color: #fcd34d;
            border: 1px solid #92400e;
        }

        .status-badge.info {
            background: #1e3a8a;
            color: #93c5fd;
            border: 1px solid #1e40af;
        }

        /* Input Fields */
        .stTextInput > div > div > input {
            background: var(--surface-elevated) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: 0.5rem !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.2) !important;
        }

        /* Select Boxes */
        .stSelectbox > div > div {
            background: var(--surface-elevated) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
        }

        /* Text Areas */
        .stTextArea textarea {
            background: var(--surface-elevated) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: 0.5rem !important;
            font-family: 'JetBrains Mono', monospace !important;
        }

        .stTextArea textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.2) !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary) 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
            padding: 0.75rem 1.5rem !important;
            box-shadow: var(--shadow-md) !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-lg) !important;
        }

        .stDownloadButton > button {
            background: var(--surface-elevated) !important;
            color: var(--primary) !important;
            border: 2px solid var(--primary) !important;
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
        }

        .stDownloadButton > button:hover {
            background: var(--primary) !important;
            color: white !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--surface-alt);
            padding: 0.375rem;
            border-radius: 0.5rem;
            border: 1px solid var(--border);
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary);
            font-weight: 600;
            border-radius: 0.375rem;
        }

        .stTabs [aria-selected="true"] {
            background: var(--surface-elevated) !important;
            color: var(--text-primary) !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background: var(--surface-elevated) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 0.5rem !important;
        }

        /* Remove Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Code blocks */
        code {
            background: var(--surface-elevated) !important;
            color: var(--primary-light) !important;
            border: 1px solid var(--border) !important;
            padding: 0.125rem 0.375rem !important;
            border-radius: 0.25rem !important;
        }
        
        /* Labels */
        .stMarkdown, label {
            color: var(--text-primary) !important;
        }
        
        /* Captions */
        .stCaptionContainer {
            color: var(--text-muted) !important;
        }
        </style>
        """
    
    else:  # Light theme
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Light Theme Variables */
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --primary-light: #818cf8;
            --secondary: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --bg-main: #f8fafc;
            --bg-secondary: #f1f5f9;
            --surface: #ffffff;
            --surface-elevated: #ffffff;
            --surface-alt: #f9fafb;
            --border: #e2e8f0;
            --border-light: #cbd5e1;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1);
        }

        /* Main Background */
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-primary);
        }

        /* Hero Section */
        .hero {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 1rem;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-xl);
            position: relative;
            overflow: hidden;
        }

        .hero::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 100%);
            pointer-events: none;
        }

        .hero-content {
            position: relative;
            z-index: 1;
        }

        .hero h1 {
            color: white;
            font-size: 2.25rem;
            font-weight: 700;
            margin: 0 0 0.5rem 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .hero p {
            color: rgba(255, 255, 255, 0.95);
            font-size: 1rem;
            margin: 0;
            line-height: 1.5;
        }

        /* Cards */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
        }

        .card:hover {
            box-shadow: var(--shadow-md);
            border-color: var(--border-light);
        }

        /* Info Cards */
        .info-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--shadow-sm);
        }

        .info-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.375rem;
        }

        .info-value {
            font-size: 0.9375rem;
            color: var(--text-primary);
            font-weight: 500;
            line-height: 1.4;
        }

        /* Metric Cards */
        .metric-item {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.25rem;
            text-align: center;
            box-shadow: var(--shadow-sm);
        }

        .metric-label {
            font-size: 0.8125rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
            line-height: 1;
        }

        .metric-suffix {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        /* Status Badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.375rem;
            padding: 0.375rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8125rem;
            font-weight: 600;
            line-height: 1;
        }

        .status-badge.success {
            background: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }

        .status-badge.warning {
            background: #fed7aa;
            color: #92400e;
            border: 1px solid #fcd34d;
        }

        .status-badge.info {
            background: #dbeafe;
            color: #1e40af;
            border: 1px solid #bfdbfe;
        }

        /* Input Fields */
        .stTextInput > div > div > input {
            background: var(--surface) !important;
            border: 2px solid var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: 0.5rem !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
        }

        /* Select Boxes */
        .stSelectbox > div > div {
            background: var(--surface) !important;
            border: 2px solid var(--border) !important;
            color: var(--text-primary) !important;
        }

        /* Text Areas */
        .stTextArea textarea {
            background: var(--surface) !important;
            border: 2px solid var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: 0.5rem !important;
            font-family: 'JetBrains Mono', monospace !important;
        }

        .stTextArea textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
            padding: 0.75rem 1.5rem !important;
            box-shadow: var(--shadow-sm) !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: var(--shadow-md) !important;
        }

        .stDownloadButton > button {
            background: var(--surface) !important;
            color: var(--primary) !important;
            border: 2px solid var(--primary) !important;
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
        }

        .stDownloadButton > button:hover {
            background: var(--primary) !important;
            color: white !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--surface-alt);
            padding: 0.375rem;
            border-radius: 0.5rem;
            border: 1px solid var(--border);
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary);
            font-weight: 600;
            border-radius: 0.375rem;
        }

        .stTabs [aria-selected="true"] {
            background: var(--surface) !important;
            color: var(--text-primary) !important;
            box-shadow: var(--shadow-sm);
        }

        /* Expander */
        .streamlit-expanderHeader {
            background: var(--surface) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 0.5rem !important;
        }

        /* Remove Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Code blocks */
        code {
            background: var(--surface-alt) !important;
            color: var(--primary-dark) !important;
            border: 1px solid var(--border) !important;
            padding: 0.125rem 0.375rem !important;
            border-radius: 0.25rem !important;
        }
        </style>
        """


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_duration(seconds: int) -> str:
    """Convert seconds to human-readable duration."""
    if seconds <= 0:
        return "0s"
    
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def estimate_reading_time(text: str) -> int:
    """Estimate reading time in minutes (200 words/min)."""
    words = len(text.split())
    return max(1, math.ceil(words / 200))


def format_number(num: int) -> str:
    """Format number with thousand separators."""
    return f"{num:,}"


def get_method_badge(method: TranscriptionMethod) -> str:
    """Get HTML badge for transcription method."""
    badges = {
        TranscriptionMethod.YOUTUBE_API: '<span class="status-badge success">✓ YouTube Captions</span>',
        TranscriptionMethod.WHISPER: '<span class="status-badge info">🤖 Whisper AI</span>',
        TranscriptionMethod.GOOGLE_SPEECH: '<span class="status-badge warning">🎤 Google Speech</span>',
    }
    return badges.get(method, '<span class="status-badge">Unknown</span>')


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        return dt.strftime('%b %d, %Y at %I:%M %p')
    except:
        return timestamp


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_hero():
    """Render hero section."""
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <h1>🎬 YouTube Transcript Studio</h1>
                <p>Professional-grade transcript extraction with multiple fallback methods. Fast, accurate, and completely free.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_theme_toggle():
    """Render theme toggle button in sidebar."""
    current_theme = st.session_state.theme
    
    if current_theme == "light":
        if st.button("🌙 Switch to Dark Mode", use_container_width=True):
            st.session_state.theme = "dark"
            st.rerun()
    else:
        if st.button("☀️ Switch to Light Mode", use_container_width=True):
            st.session_state.theme = "light"
            st.rerun()


def render_dependency_status():
    """Render dependency status in sidebar."""
    with st.expander("📦 Dependency Status", expanded=False):
        deps = DependencyChecker.check_all()
        
        for name, available in deps.items():
            if available:
                st.markdown(f"✅ **{name}**")
            else:
                st.markdown(f"❌ **{name}** (optional)")
        
        st.caption("Install missing dependencies with:")
        st.code("pip install -r requirements.txt", language="bash")


def render_info_cards():
    """Render information cards about the transcription process."""
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Transcription Flow</div>
            <div class="info-value">YouTube Captions → Whisper AI → Google Speech API</div>
        </div>
        <div class="info-card">
            <div class="info-label">Supported Formats</div>
            <div class="info-value">TXT • SRT • VTT • JSON</div>
        </div>
        <div class="info-card">
            <div class="info-label">Features</div>
            <div class="info-value">Intelligent caching • Retry logic • Multi-language</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(result: TranscriptResult):
    """Render metrics cards."""
    word_count = len(result.full_text.split())
    char_count = len(result.full_text)
    reading_time = estimate_reading_time(result.full_text)
    duration = result.video_info.duration
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-item">
                <div class="metric-label">Words</div>
                <div class="metric-value">{format_number(word_count)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-item">
                <div class="metric-label">Characters</div>
                <div class="metric-value">{format_number(char_count)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-item">
                <div class="metric-label">Reading Time</div>
                <div class="metric-value">{reading_time}</div>
                <div class="metric-suffix">minutes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="metric-item">
                <div class="metric-label">Video Duration</div>
                <div class="metric-value">{format_duration(duration)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_video_info(result: TranscriptResult):
    """Render video information card."""
    st.markdown(
        f"""
        <div class="card">
            <div style="margin-bottom: 1rem;">
                <strong style="color: var(--text-primary); font-size: 1.125rem;">Video Information</strong>
            </div>
            <div style="margin-top: 1rem;">
                <p style="margin: 0.5rem 0; color: var(--text-primary);"><strong>Title:</strong> {result.video_info.title}</p>
                <p style="margin: 0.5rem 0; color: var(--text-primary);"><strong>Author:</strong> {result.video_info.author}</p>
                <p style="margin: 0.5rem 0; color: var(--text-primary);"><strong>Duration:</strong> {format_duration(result.video_info.duration)}</p>
                <p style="margin: 0.5rem 0; color: var(--text-primary);"><strong>Video ID:</strong> <code>{result.video_info.video_id}</code></p>
                <p style="margin: 0.5rem 0; color: var(--text-primary);"><strong>Transcribed:</strong> {format_timestamp(result.timestamp)}</p>
                <p style="margin: 0.5rem 0;"><strong style="color: var(--text-primary);">Method:</strong> {get_method_badge(result.method)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_transcript_tabs(result: TranscriptResult):
    """Render transcript tabs with preview and full text."""
    tab1, tab2, tab3 = st.tabs(["📄 Preview", "📖 Full Transcript", "⏱️ Segments"])
    
    with tab1:
        preview_text = result.full_text[:3000]
        if len(result.full_text) > 3000:
            preview_text += "\n\n... (preview truncated)"
        
        st.text_area(
            "Transcript Preview",
            value=preview_text,
            height=400,
            help="First 3000 characters of the transcript",
            label_visibility="collapsed",
        )
    
    with tab2:
        st.text_area(
            "Full Transcript",
            value=result.full_text,
            height=500,
            label_visibility="collapsed",
        )
    
    with tab3:
        st.caption(f"Total segments: {len(result.segments)}")
        
        # Show first 20 segments with timestamps
        display_segments = result.segments[:20]
        
        for i, segment in enumerate(display_segments, 1):
            start_time = format_duration(int(segment.start))
            end_time = format_duration(int(segment.end()))
            
            st.markdown(
                f"""
                <div style="padding: 0.75rem; background: var(--surface-elevated); 
                     border: 1px solid var(--border); border-radius: 0.5rem; margin-bottom: 0.5rem;">
                    <div style="font-size: 0.75rem; color: var(--text-muted); 
                         font-family: 'JetBrains Mono', monospace; margin-bottom: 0.25rem;">
                        [{start_time} → {end_time}]
                    </div>
                    <div style="color: var(--text-primary);">{segment.text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        if len(result.segments) > 20:
            st.info(f"Showing first 20 of {len(result.segments)} segments. Download the full transcript to see all.")


def render_download_section(result: TranscriptResult, output_filename: str):
    """Render download buttons for different formats."""
    st.markdown("### 📥 Download Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # TXT format
    with col1:
        from youtube_transcriber_improved import TranscriptFormatter
        txt_content = TranscriptFormatter.to_text(result, include_metadata=True)
        st.download_button(
            label="📄 TXT",
            data=txt_content,
            file_name=f"{Path(output_filename).stem}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    
    # SRT format
    with col2:
        srt_content = TranscriptFormatter.to_srt(result)
        st.download_button(
            label="🎬 SRT",
            data=srt_content,
            file_name=f"{Path(output_filename).stem}.srt",
            mime="text/plain",
            use_container_width=True,
        )
    
    # VTT format
    with col3:
        vtt_content = TranscriptFormatter.to_vtt(result)
        st.download_button(
            label="📹 VTT",
            data=vtt_content,
            file_name=f"{Path(output_filename).stem}.vtt",
            mime="text/vtt",
            use_container_width=True,
        )
    
    # JSON format
    with col4:
        json_content = TranscriptFormatter.to_json(result, pretty=True)
        st.download_button(
            label="🔧 JSON",
            data=json_content,
            file_name=f"{Path(output_filename).stem}.json",
            mime="application/json",
            use_container_width=True,
        )
    
    st.caption("💡 **TXT** = Plain text • **SRT/VTT** = Subtitles with timestamps • **JSON** = Structured data")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application logic."""
    init_session_state()
    
    # Apply theme CSS
    st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)
    
    # Hero Section
    render_hero()
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Theme Toggle
        render_theme_toggle()
        
        st.markdown("---")
        
        # Language Selection
        language_options = {
            "English (US)": "en",
            "English (UK)": "en-GB",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese (Simplified)": "zh",
            "Hindi": "hi",
            "Arabic": "ar",
        }
        
        selected_language = st.selectbox(
            "Language",
            options=list(language_options.keys()),
            index=0,
            help="Language for caption lookup and transcription",
        )
        language = language_options[selected_language]
        
        # Output Format
        output_format = st.selectbox(
            "Primary Output Format",
            options=["TXT", "SRT", "VTT", "JSON"],
            index=0,
            help="Primary format for saved transcript",
        )
        
        # Advanced Options
        with st.expander("🔧 Advanced Options", expanded=False):
            force_method = st.selectbox(
                "Force Transcription Method",
                options=["Auto (Recommended)", "YouTube Captions Only", "Whisper AI", "Google Speech"],
                index=0,
                help="Override automatic method selection",
            )
            
            whisper_model = st.selectbox(
                "Whisper Model",
                options=["tiny", "base", "small", "medium"],
                index=1,
                help="Larger models = better quality but slower",
            )
            
            enable_cache = st.checkbox(
                "Enable Caching",
                value=True,
                help="Cache transcripts to avoid re-processing",
            )
            
            # Update config
            st.session_state.config = TranscriberConfig(
                enable_cache=enable_cache,
                whisper_model=whisper_model,
            )
        
        st.markdown("---")
        render_dependency_status()
        
        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit")
        st.caption("Powered by yt-dlp, Whisper, & YouTube API")
    
    # Main Content Area
    col_left, col_right = st.columns([1.5, 1], gap="large")
    
    with col_left:
        st.markdown("### 🎯 Enter YouTube URL")
        
        url = st.text_input(
            "YouTube URL or Video ID",
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            help="Paste any YouTube video URL or just the video ID",
            label_visibility="collapsed",
        )
        
        col_btn1, col_btn2 = st.columns([3, 1])
        
        with col_btn1:
            generate_clicked = st.button(
                "🚀 Generate Transcript",
                use_container_width=True,
                type="primary",
            )
        
        with col_btn2:
            if st.session_state.transcript_result:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.transcript_result = None
                    st.rerun()
    
    with col_right:
        render_info_cards()
    
    # Processing Logic
    if generate_clicked and url.strip():
        # Check dependencies
        missing_core = DependencyChecker.check_core()
        if missing_core:
            st.error(
                f"❌ Missing required dependencies: **{', '.join(missing_core)}**\n\n"
                f"Install with: `pip install {' '.join(missing_core)}`"
            )
            return
        
        # Create transcriber
        transcriber = YouTubeTranscriber(st.session_state.config)
        
        # Determine method
        force_method_map = {
            "Auto (Recommended)": None,
            "YouTube Captions Only": TranscriptionMethod.YOUTUBE_API,
            "Whisper AI": TranscriptionMethod.WHISPER,
            "Google Speech": TranscriptionMethod.GOOGLE_SPEECH,
        }
        method = force_method_map.get(force_method)
        
        # Process with progress indicator
        with st.spinner("🔄 Processing transcript... This may take a few moments."):
            try:
                result = transcriber.transcribe(
                    youtube_url=url.strip(),
                    language=language,
                    force_method=method,
                )
                
                if result:
                    st.session_state.transcript_result = result
                    st.session_state.last_url = url.strip()
                    st.success(f"✅ Transcript generated successfully using **{result.method.value}**!")
                    st.rerun()
                else:
                    st.error(
                        "❌ Transcription failed. Please check:\n"
                        "- Video URL is valid\n"
                        "- Video has captions or audio\n"
                        "- All dependencies are installed"
                    )
            
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)
    
    elif generate_clicked:
        st.warning("⚠️ Please enter a YouTube URL")
    
    # Display Results
    if st.session_state.transcript_result:
        st.markdown("---")
        st.markdown("## 📊 Results")
        
        result = st.session_state.transcript_result
        
        # Metrics
        render_metrics(result)
        
        # Video Info and Transcript
        col_info, col_transcript = st.columns([1, 2], gap="large")
        
        with col_info:
            render_video_info(result)
        
        with col_transcript:
            render_transcript_tabs(result)
        
        # Download Section
        st.markdown("---")
        output_filename = f"transcript_{result.video_info.video_id}"
        render_download_section(result, output_filename)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()