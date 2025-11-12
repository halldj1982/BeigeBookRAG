import streamlit as st
from utils import load_config
from opensearch_vector_store import OpenSearchVectorStore

st.set_page_config(page_title="Browse Documents", page_icon="📚")
st.title("📚 Browse Vector Database")

cfg = load_config()

@st.cache_resource
def get_vector_store():
    return OpenSearchVectorStore(cfg)

vector_store = get_vector_store()

if st.button("🔄 Load 5 Random Entries"):
    try:
        results = vector_store.client.search(
            index=cfg["opensearch_index"],
            body={
                "size": 5,
                "query": {
                    "function_score": {
                        "query": {"match_all": {}},
                        "random_score": {}
                    }
                },
                "_source": {"excludes": ["embedding"]}
            }
        )
        
        if results['hits']['total']['value'] == 0:
            st.warning("No documents found in the vector database.")
        else:
            st.success(f"Total documents in database: {results['hits']['total']['value']}")
            
            for i, hit in enumerate(results['hits']['hits'], 1):
                with st.expander(f"Document {i} - {hit['_source'].get('source', 'Unknown')}"):
                    metadata = hit['_source']
                    st.json(metadata)
                    
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
