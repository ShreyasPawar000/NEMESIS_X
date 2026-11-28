import streamlit as st

def inject_css(path):
    st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def metric_card(column, label, value, detail):
    column.markdown(
        f'<div class="metric"><small>{label}</small><strong>{value}</strong><span>{detail}</span></div>',
        unsafe_allow_html=True
    )

def section_heading(kicker, title):
    st.markdown(f'<div class="section-head"><small>{kicker}</small><h2>{title}</h2></div>', unsafe_allow_html=True)
