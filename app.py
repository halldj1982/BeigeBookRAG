import streamlit as st

st.set_page_config(
    page_title="BeigeBot — Home",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 BeigeBot")
st.subheader("Your Personal Beige Book Assistant")
st.write(
    "BeigeBot uses Retrieval-Augmented Generation (RAG) to answer your questions about Federal Reserve Beige Books "
    "with intelligent document chunking, metadata extraction, and semantic search powered by Amazon Bedrock and OpenSearch."
)
st.write("### 📑 Available Pages")

st.write(
    "**1. Chatbot** — Ask questions about the Federal Reserve Beige Books. "
    "The system performs k-NN vector search in OpenSearch and generates answers using Amazon Nova Premier (us.amazon.nova-premier-v1:0). "
    "Responses include numbered citations with district, topic, and date metadata."
)

st.write(
    "**2. Ingest** — Upload Beige Book PDFs or text files. The system intelligently parses document structure, "
    "extracts metadata (districts, topics, publication dates), creates semantic chunks, "
    "generates embeddings using Amazon Titan Text Embeddings v2 (1024 dimensions), "
    "and indexes into OpenSearch with k-NN vector support."
)

st.write(
    "**3. Browse** — View random samples from the vector database to inspect stored documents and their metadata. "
    "Useful for verifying ingestion quality and exploring the knowledge base structure."
)
st.write("### 🏗️ Technology Stack")

st.write(
    "- **LLM**: Amazon Nova Premier (us.amazon.nova-premier-v1:0)\n"
    "- **Embeddings**: Amazon Titan Text Embeddings v2 (1024-dim)\n"
    "- **Vector Store**: AWS OpenSearch with k-NN plugin\n"
    "- **Infrastructure**: Terraform-managed VPC, EC2, and OpenSearch domain\n"
    "- **Document Processing**: PDFMiner + custom Beige Book parser\n"
    "- **Frontend**: Streamlit on EC2"
)
