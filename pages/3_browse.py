import streamlit as st
from utils import load_config
from opensearch_vector_store import OpenSearchVectorStore
import re

st.set_page_config(page_title="BeigeBot Browse", page_icon="🤖")
st.title("🤖 BeigeBot Knowledge Base Browser")

cfg = load_config()

@st.cache_resource
def get_vector_store():
    return OpenSearchVectorStore(cfg)

vector_store = get_vector_store()

st.write("Filter documents by metadata:")

col1, col2, col3 = st.columns(3)

with col1:
    requested_beigebook = st.text_input("Beige Book (YYYYMM)", placeholder="202510", help="Format: YYYYMM (e.g., 202510 for October 2025)")

with col2:
    district = st.text_input("District", placeholder="Atlanta", help="Federal Reserve district name")

with col3:
    section_type = st.selectbox("Section Type", ["All", "national_summary", "district_report", "about"])

max_results = st.slider("Max results to display", 5, 100, 20)

if st.button("🔍 Search Documents", type="primary"):
    try:
        # Build query with filters
        must_filters = []
        
        if requested_beigebook:
            # Extract YYYYMM from source filename
            must_filters.append({
                "wildcard": {
                    "source": f"*{requested_beigebook}*"
                }
            })
        
        if district:
            must_filters.append({
                "term": {
                    "district": district
                }
            })
        
        if section_type and section_type != "All":
            must_filters.append({
                "term": {
                    "section_type": section_type
                }
            })
        
        query_body = {
            "size": max_results,
            "query": {
                "bool": {
                    "must": must_filters if must_filters else [{"match_all": {}}]
                }
            },
            "_source": {"excludes": ["embedding"]}
        }
        
        results = vector_store.client.search(
            index=cfg["opensearch_index"],
            body=query_body
        )
        
        total = results['hits']['total']['value']
        hits = results['hits']['hits']
        
        if total == 0:
            st.warning("No documents found matching the filters.")
        else:
            st.success(f"Found {total} documents (showing {len(hits)})")
            
            for i, hit in enumerate(hits, 1):
                source_data = hit['_source']
                title = f"{source_data.get('source', 'Unknown')}"
                if source_data.get('district'):
                    title += f" - {source_data['district']}"
                if source_data.get('section_type'):
                    title += f" ({source_data['section_type']})"
                
                with st.expander(f"Document {i}: {title}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Metadata:**")
                        st.write(f"Source: {source_data.get('source', 'N/A')}")
                        st.write(f"District: {source_data.get('district', 'N/A')}")
                        st.write(f"Section: {source_data.get('section_type', 'N/A')}")
                        st.write(f"Topic: {source_data.get('topic', 'N/A')}")
                    with col_b:
                        st.write(f"Heading: {source_data.get('heading', 'N/A')}")
                        st.write(f"Chunk Index: {source_data.get('chunk_index', 'N/A')}")
                        st.write(f"Word Count: {source_data.get('word_count', 'N/A')}")
                    
                    st.write("**Text:**")
                    st.text_area("Content", source_data.get('text', ''), height=200, key=f"text_{i}")
                    
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        st.exception(e)
