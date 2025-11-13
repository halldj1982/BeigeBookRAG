import streamlit as st
from ingest import Ingestor
from utils import load_config
import tempfile

st.set_page_config(page_title="BeigeBot Ingest", page_icon="🤖", layout="wide")
st.title("🤖 BeigeBot Document Ingestion")

config = load_config()
ingestor = Ingestor(config)

st.subheader("⚠️ Danger Zone")
if st.button("🗑️ Wipe Knowledge Base", type="secondary"):
    with st.spinner("Deleting index and recreating..."):
        try:
            ingestor.vs.delete_index()
            st.success("Knowledge base wiped successfully! Index recreated with new schema.")
        except Exception as e:
            st.error(f"Error wiping knowledge base: {e}")

st.divider()

uploaded = st.file_uploader("Upload PDF or TXT files", type=['pdf', 'txt'], accept_multiple_files=True)
if uploaded:
    for up in uploaded:
        with st.spinner(f"Processing {up.name}..."):
            try:
                if up.name.endswith('.txt'):
                    text = up.read().decode('utf-8')
                    result = ingestor.ingest_text(text, source_name=up.name)
                else:
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    tf.write(up.read())
                    tf.flush()
                    tf.close()
                    result = ingestor.ingest_pdf(tf.name, source_name=up.name)
                st.success(f"Indexed {up.name} — created {result['num_docs']} documents from {result['num_chunks']} chunks.")
            except Exception as e:
                st.error(f"Error processing {up.name}: {e}")

st.markdown("Upload documents to expand BeigeBot's knowledge base.")
