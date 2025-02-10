import streamlit as st
import streamlit.components.v1 as components

from engine import NERProcessingEngine

st.set_page_config(layout="wide")
st.title("Entity Extraction Dashboard")


@st.cache_resource
def load_model():
    engine = NERProcessingEngine("data/ProjectPhoenixPlan.md", devMode=False)
    engine.pipeline(html_height="100vh", html_width="100%")
    return engine


st.sidebar.title("Control Panel")
st.sidebar.write("Upload your data file in the following component.")
st.sidebar.markdown("---")
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.write("File uploaded successfully!")

col1, col2 = st.columns(2)
with col1:
    with st.expander("Extracted Entities", expanded=True):
        with st.container():
            engine = load_model()
            ent_html = engine.render_entities()
            st.markdown(
                f'<div style="height: 525px; overflow-x: hidden; overflow-y: scroll;">{ent_html}</div>',
                unsafe_allow_html=True,
            )
with col2:
    with st.expander("Graph Representation", expanded=True):
        with st.container():
            components.html(engine.graph_html, height=500, width=None, scrolling=False)
