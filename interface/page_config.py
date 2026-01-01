import streamlit as st

def setup_page():
    """Config page settings"""
    
    st.set_page_config(
        page_title="RL Playground",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "# RL Playground\nReinforcement Learning Visualizer"
        }
    )
    
    # Force light theme
    st.markdown("""
        <script>
        window.parent.document.body.style.backgroundColor = '#ffffff';
        </script>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Layout */
        .main {
            background-color: #ffffff;
            color: #1e293b;
        }
        
        .main .block-container {
            padding: 2rem 3rem;
            max-width: 1400px;
        }
        
        /* Typography */
        h1 {
            color: #0f172a !important;
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            margin-bottom: 1rem !important;
            letter-spacing: -0.02em !important;
        }
        
        h2 {
            color: #1e293b !important;
            font-weight: 600 !important;
            font-size: 1.875rem !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }
        
        h3 {
            color: #334155 !important;
            font-weight: 600 !important;
            font-size: 1.5rem !important;
            margin-top: 1.5rem !important;
        }
        
        h4 {
            color: #475569 !important;
            font-weight: 500 !important;
            font-size: 1.25rem !important;
        }
        
        p, span, div, label {
            color: #475569 !important;
            line-height: 1.6 !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #334155 0%, #1e293b 100%);
            border-right: 1px solid #475569;
        }
        
        [data-testid="stSidebar"] * {
            color: #f1f5f9 !important;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stSidebar"] label {
            color: #e2e8f0 !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 12px rgba(16, 185, 129, 0.3);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Button variants */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }
        
        .stButton > button[kind="secondary"] {
            background: linear-gradient(135deg, #64748b 0%, #475569 100%);
            box-shadow: 0 4px 6px rgba(100, 116, 139, 0.2);
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #475569 0%, #334155 100%);
        }
        
        /* UI Components */
        .stSlider > div > div > div {
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        }
        
        .stSlider > div > div > div > div {
            background-color: #ffffff;
            border: 3px solid #10b981;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
        }
        
        .stSelectbox > div > div {
            border-radius: 8px;
            border: 2px solid #e2e8f0;
            transition: all 0.2s ease;
        }
        
        .stSelectbox > div > div:hover {
            border-color: #10b981;
        }
        
        .stSelectbox > div > div:focus-within {
            border-color: #10b981;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: transparent;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            color: #64748b !important;
            background: transparent;
            border: none;
            transition: all 0.2s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #f1f5f9;
            color: #10b981 !important;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white !important;
            box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #10b981 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Notifications */
        .stAlert {
            border-radius: 12px;
            border-left: 4px solid;
            padding: 1rem 1.25rem;
        }
        
        .stAlert[data-baseweb="notification"][kind="info"] {
            background: #f0fdf4;
            border-left-color: #10b981;
        }
        
        .stAlert[data-baseweb="notification"][kind="warning"] {
            background: #fffbeb;
            border-left-color: #f59e0b;
        }
        
        .stAlert[data-baseweb="notification"][kind="success"] {
            background: #f0fdf4;
            border-left-color: #10b981;
        }
        
        /* Containers */
        .streamlit-expanderHeader {
            background-color: #f8fafc;
            border-radius: 8px;
            font-weight: 600;
            color: #1e293b !important;
            border: 1px solid #e2e8f0;
            transition: all 0.2s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #f0fdf4;
            border-color: #10b981;
        }
        
        .stProgress > div > div {
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            border-radius: 10px;
        }
        
        /* Animations */
        @keyframes pulse-color {
            0%, 100% {
                opacity: 1;
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            }
            50% {
                opacity: 0.8;
                box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
            }
        }
        
        .status-indicator {
            animation: pulse-color 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        /* Downloads */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #64748b 0%, #475569 100%);
            color: white !important;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            font-size: 0.875rem;
            transition: all 0.2s ease;
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #475569 0%, #334155 100%);
            transform: translateY(-1px);
        }
        
        /* Tables */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }
        
        .dataframe th {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white !important;
            font-weight: 600;
            padding: 0.75rem;
        }
        
        .dataframe td {
            padding: 0.75rem;
            border-bottom: 1px solid #f1f5f9;
        }
        
        .dataframe tr:hover {
            background-color: #f0fdf4;
        }
        
        /* Images */
        .stImage {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .stImage:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        
        /* Mobile */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
            }
            
            h1 {
                font-size: 2rem !important;
            }
            
            h2 {
                font-size: 1.5rem !important;
            }
        }
        
        /* Global tweaks */
        * {
            transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
        }
        
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #10b981 0%, #059669 100%);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #059669 0%, #047857 100%);
        }
        
        .caption {
            color: #64748b !important;
            font-size: 0.875rem !important;
            font-style: italic !important;
        }
        
        .element-container {
            transition: all 0.3s ease;
        }
        
        .stSuccess {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-left: 4px solid #10b981;
            border-radius: 8px;
        }
        
        .stError {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border-left: 4px solid #ef4444;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)
