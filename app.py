import streamlit as st

st.set_page_config(
    page_title="BeigeBook RAG — Home",
    page_icon="📊",
    layout="wide",
)

st.title("📊 BeigeBook RAG Chatbot")
st.subheader("Powered by Amazon Bedrock + OpenSearch Vector DB")

st.write("### Welcome to the Beige Book Retrieval-Augmented Chatbot")

st.write(
    "- Use the sidebar on the left to navigate between pages."
)
st.write(
    "1. **Chatbot** — Ask questions about the Federal Reserve Beige Books. "
    "System performs vector search in OpenSearch and generates answers with Claude Sonnet 3.7."
)
st.write(
    "2. **Ingest PDFs** — Upload Beige Book PDFs. The system extracts text, chunks it, "
    "embeds using Bedrock, and loads into OpenSearch as vector embeddings."
)

st.write("---")
st.write("### 🔐 Required Streamlit Secrets (set in Streamlit Community Cloud)")

st.code(
    'aws_region = "us-east-1"\n'
    'opensearch_endpoint = "search-xxxxx.us-east-1.es.amazonaws.com"\n'
    'opensearch_index = "beigebook-docs"\n'
    'claude_model = "anthropic.claude-sonnet-3.7"\n'
    'bedrock_embedding_model =')
