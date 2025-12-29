"""Main content area with tabs"""

import streamlit as st
from interface.tabs import render_training_tab, render_inference_tab, render_design_tab


def render_main_content():
    """Render main content with tabs"""
    
    tab1, tab2, tab3 = st.tabs(["Training", "Inference", "Design & Analysis"])
    
    with tab1:
        render_training_tab()
    
    with tab2:
        render_inference_tab()
    
    with tab3:
        render_design_tab()
