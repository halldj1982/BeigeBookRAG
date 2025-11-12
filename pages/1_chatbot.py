import streamlit as st
from rag_opensearch import RAGOpenSearch
from utils import load_config

st.set_page_config(page_title="BeigeBook Chatbot", layout="wide")
st.title("BeigeBook Assistant — Chatbot")
st.write("Compare responses with and without RAG context. Powered by Amazon Nova Premier via Bedrock and OpenSearch vector store.")

config = load_config()
rag = RAGOpenSearch(config)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("top_k", 1, 20, 5)
    rerank_threshold = st.slider("Rerank threshold", 0.0, 1.0, 0.65)
    
    st.divider()
    if st.button("🆕 New Chat", type="primary", use_container_width=True):
        # Clear all session state
        st.session_state.history = []
        if 'last_meta' in st.session_state:
            del st.session_state['last_meta']
        st.rerun()

query = st.text_input("Ask about the Beige Book (community development examples encouraged)")
if st.button("Ask") and query.strip():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Standard Response (No RAG)")
        with st.spinner("Generating standard response..."):
            # Get standard response without RAG
            standard_resp = rag.bedrock.generate(prompt=query)
            standard_answer = standard_resp.get('output', 'No response')
        st.markdown(standard_answer)
    
    with col2:
        st.subheader("📚 RAG Response (With Context)")
        with st.spinner("Retrieving context and generating..."):
            # Get RAG response
            rag_resp = rag.answer(query=query, top_k=top_k, rerank_threshold=rerank_threshold, history=st.session_state.history)
            rag_answer = rag_resp['answer']
        st.markdown(rag_answer)
        
        # Show retrieved context
        with st.expander("📖 Retrieved Context"):
            meta = rag_resp.get('meta', {})
            if 'retrieved_docs' in meta:
                for i, doc in enumerate(meta['retrieved_docs'][:3]):
                    st.write(f"**Source {i+1}:** {doc.get('source', 'Unknown')} (Page {doc.get('page', 'N/A')})")
                    st.write(f"*Score: {doc.get('score', 'N/A')}*")
                    st.write(doc.get('text', '')[:200] + "...")
                    st.divider()
    
    # Store both responses in history
    st.session_state.history.append({"role":"user","content":query})
    st.session_state.history.append({"role":"assistant","content":f"**Standard:** {standard_answer}\n\n**RAG:** {rag_answer}"})
    st.session_state['last_meta'] = rag_resp.get('meta', {})

if st.session_state.history:
    st.subheader("💬 Conversation History")
    for t in st.session_state.history[-10:]:
        if t['role']=='user':
            st.markdown(f"**You:** {t['content']}")
        else:
            st.markdown(f"**Assistant:** {t['content']}")
            st.divider()

with st.sidebar:
    st.subheader("Last retrieval metadata")
    st.json(st.session_state.get('last_meta', {}))
