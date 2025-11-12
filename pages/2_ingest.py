import streamlit as st
from ingest import Ingestor
from utils import load_config
import tempfile

st.set_page_config(page_title="Ingest PDFs", layout="wide")
st.title("Upload & Ingest Beige Book PDFs")

config = load_config()
ingestor = Ingestor(config)

uploaded = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
if uploaded:
    for up in uploaded:
        with st.spinner(f"Processing {up.name}..."):
            # save to temp and ingest
            tf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            tf.write(up.read())
            tf.flush()
            tf.close()
            result = ingestor.ingest_pdf(tf.name, source_name=up.name)
            st.success(f"Indexed {result['num_pages']} pages from {up.name} — created {result['num_docs']} documents.")
st.markdown("Make sure OpenSearch endpoint and Bedrock embedding model are configured in secrets.")
