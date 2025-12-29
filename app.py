import streamlit as st

from interface.page_config import setup_page
from interface.sidebar import render_sidebar
from interface.main_content import render_main_content

def main():
    """Main app entry point"""
    
    setup_page()
    
    if 'selected_environment' not in st.session_state:
        st.session_state.selected_environment = 'GridWorld'
    if 'selected_algorithm' not in st.session_state:
        st.session_state.selected_algorithm = 'Q-Learning'
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'trained_agent' not in st.session_state:
        st.session_state.trained_agent = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = None
    
    st.markdown("""
        <div style='text-align: center; padding: 2.5rem 0 1.5rem 0; border-bottom: 3px solid #10b981;'>
            <h1 style='
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                letter-spacing: -0.02em;'>
                Reinforcement Learning Playground
            </h1>
            <p style='
                color: #64748b;
                font-size: 1.125rem;
                font-weight: 500;
                margin-top: 0;
                letter-spacing: 0.02em;'>
                Interactive Tool for Mastering RL Algorithms
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    render_sidebar()
    render_main_content()
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #94a3b8; padding: 1.5rem; font-size: 0.875rem;'>
            <p style='margin: 0;'>
                <strong style='color: #10b981;'>RL Playground</strong> | 
                Built for RL Learners | 
                <span style='color: #64748b;'>Explore - Learn - Master</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
