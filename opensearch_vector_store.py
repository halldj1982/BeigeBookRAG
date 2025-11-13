# opensearch_vector_store.py
"""
Robust OpenSearch vector store helper (drop-in replacement).

Features:
- Builds an AWS4-signed OpenSearch client (auto-detects region, includes session token).
- Ensures index with k-NN vector mapping (dimension taken from config['embed_dim']).
- index_document(doc_id, text, embedding, source, page)
- search(query, top_k): if passed a string, will try to use bedrock_client.BedrockClient to embed the query.
- search_with_embedding(embedding, top_k): performs k-NN search using OpenSearch knn query.
- Defensive handling for opensearch-py versions that require keyword args.
"""

from urllib.parse import urlparse
import re
import sys
import os
from typing import List, Dict, Any

import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection, exceptions as os_exceptions

# Try import BedrockClient if available in the project; used for embedding queries in search()
try:
    from bedrock_client import BedrockClient  # optional
    _HAS_BEDROCK_CLIENT = True
except Exception:
    _HAS_BEDROCK_CLIENT = False


class OpenSearchVectorStore:
    def __init__(self, config: Dict[str, Any]):
        """
        config should contain:
          - opensearch_endpoint (full URL or host)
          - opensearch_index (optional, default 'beigebook-docs')
          - aws_region (optional, will try boto3 session or parse from endpoint)
          - embed_dim (required when creating the index)
        """
        self.config = config or {}
        endpoint = self.config.get("opensearch_endpoint")
        if not endpoint:
            raise RuntimeError("OpenSearch endpoint not configured. Set opensearch_endpoint in secrets or env.")

        # Normalize endpoint -> host
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            parsed = urlparse(endpoint)
            host = parsed.hostname
        else:
            host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")

        # Determine region: config -> boto3 session -> parse from endpoint host
        region = self.config.get("aws_region") or boto3.Session().region_name
        if not region:
            # try to infer region from host (common aws hosted opensearch hostnames)
            m = re.search(r"\.([a-z0-9-]+-[0-9])\.es\.amazonaws\.com$", host)
            if m:
                # e.g., search-domain-abc.us-west-2.es.amazonaws.com -> extract us-west-2
                # fallback regex: match us-west-2 etc
                m2 = re.search(r"(us|ap|eu|sa|ca|me|af|cn)-[a-z0-9-]+-[0-9]", host)
                if m2:
                    region = m2.group(0)

        if not region:
            raise RuntimeError(
                "Could not determine AWS region for signing. Set 'aws_region' in secrets or AWS_REGION env var."
            )

        # Acquire credentials (may include session token)
        session = boto3.Session()
        creds = session.get_credentials()
        if creds is None:
            raise RuntimeError("No AWS credentials found. Run 'aws configure' or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY")

        frozen = creds.get_frozen_credentials()
        access_key = frozen.access_key
        secret_key = frozen.secret_key
        token = frozen.token  # may be None

        # Diagnostics to stderr (helpful when running Streamlit)
        print(
            f"[OpenSearchVectorStore] host={host} region={region} access_key_set={bool(access_key)} token_present={bool(token)}",
            file=sys.stderr,
        )

        awsauth = AWS4Auth(access_key, secret_key, region, "es", session_token=token)

        # Build OpenSearch client
        self.client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

        # index name and embed dim
        self.index_name = self.config.get("opensearch_index", "beigebook-docs")
        self.embed_dim = int(self.config.get("embed_dim", 1536))

        # ensure index exists (mapping with knn_vector)
        self._ensure_index()

    def _ensure_index(self):
        # Check index existence in a robust way
        exists = False
        try:
            # prefer keyword argument 'index' (compatible with newer clients)
            exists = self.client.indices.exists(index=self.index_name)
        except TypeError:
            # fallback: low-level HEAD request
            try:
                resp = self.client.transport.perform_request("HEAD", f"/{self.index_name}")
                # opensearch-py perform_request returns (status, headers, data) sometimes; handle both forms
                if isinstance(resp, tuple):
                    status = resp[0]
                else:
                    # some implementations return a requests.Response-like object
                    status = getattr(resp, "status_code", None)
                exists = status is not None and 200 <= int(status) < 300
            except Exception:
                exists = False
        except os_exceptions.NotFoundError:
            exists = False
        except Exception:
            # final fallback: try get
            try:
                self.client.indices.get(index=self.index_name)
                exists = True
            except Exception:
                exists = False

        if not exists:
            mapping = {
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "source": {"type": "keyword"},
                        "embedding": {"type": "knn_vector", "dimension": self.embed_dim},
                        "publication_date": {"type": "keyword"},
                        "district": {"type": "keyword"},
                        "district_number": {"type": "integer"},
                        "section_type": {"type": "keyword"},
                        "topic": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "heading": {"type": "text"},
                        "word_count": {"type": "integer"},
                    }
                },
            }
            # create index
            try:
                self.client.indices.create(index=self.index_name, body=mapping)
                print(f"[OpenSearchVectorStore] Created index '{self.index_name}' with embed_dim={self.embed_dim}", file=sys.stderr)
            except Exception as e:
                # If creation fails, show error and re-raise
                print(f"[OpenSearchVectorStore] Failed to create index: {e}", file=sys.stderr)
                raise

    def index_document(self, doc_id: str, text: str, embedding: List[float], metadata: Dict[str, Any] = None):
        """
        Index a document chunk with a precomputed embedding and metadata.
        """
        # Validate embedding
        if not embedding or not isinstance(embedding, list) or len(embedding) == 0:
            raise ValueError(f"Invalid embedding: {type(embedding)} with length {len(embedding) if embedding else 'None'}")
        
        if len(embedding) != self.embed_dim:
            raise ValueError(f"Embedding dimension mismatch: got {len(embedding)}, expected {self.embed_dim}")
        
        metadata = metadata or {}
        body = {
            "text": text,
            "embedding": embedding,
            "source": metadata.get("source", "unknown"),
            "publication_date": metadata.get("publication_date"),
            "district": metadata.get("district"),
            "district_number": metadata.get("district_number"),
            "section_type": metadata.get("section_type"),
            "topic": metadata.get("topic"),
            "chunk_index": metadata.get("chunk_index"),
            "heading": metadata.get("heading"),
            "word_count": metadata.get("word_count"),
        }
        
        # Check for dimension mismatch and recreate index if needed
        try:
            mapping = self.client.indices.get_mapping(index=self.index_name)
            current_dim = mapping[self.index_name]['mappings']['properties']['embedding']['dimension']
            if current_dim != self.embed_dim:
                import streamlit as st
                st.warning(f"Dimension mismatch! Index has {current_dim}, config has {self.embed_dim}. Recreating index...")
                self.client.indices.delete(index=self.index_name)
                self._ensure_index()
                st.success("Index recreated with correct dimensions!")
        except Exception:
            pass  # Index might not exist yet
        
        try:
            self.client.index(index=self.index_name, id=doc_id, body=body, refresh=True)
        except Exception as e:
            st.error(f"OpenSearch index error: {e}")
            st.error(f"Embedding type: {type(embedding)}, length: {len(embedding) if embedding else 'None'}")
            raise

    def search_with_embedding(self, embedding: List[float], top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform k-NN search using an embedding vector with optional pre-filtering.
        Returns list of hits with keys: id, score, text, source, district, section_type, etc.
        """
        # Build query with optional filters
        if filters:
            filter_clauses = []
            if filters.get('source'):
                filter_clauses.append({"wildcard": {"source": filters['source']}})
            if filters.get('district'):
                filter_clauses.append({"term": {"district": filters['district']}})
            if filters.get('section_type'):
                filter_clauses.append({"term": {"section_type": filters['section_type']}})
            
            body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": top_k,
                                }
                            }
                        },
                        "filter": filter_clauses
                    }
                }
            }
        else:
            body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": embedding,
                            "k": top_k,
                        }
                    }
                }
            }
        try:
            res = self.client.search(index=self.index_name, body=body)
        except Exception as e:
            print(f"[OpenSearchVectorStore] search_with_embedding error: {e}", file=sys.stderr)
            raise

        hits = []
        for h in res.get("hits", {}).get("hits", []):
            score = float(h.get("_score", 0.0))
            src = h.get("_source", {})
            hits.append(
                {
                    "id": h.get("_id"),
                    "score": score,
                    "text": src.get("text", ""),
                    "source": src.get("source", "unknown"),
                    "district": src.get("district"),
                    "section_type": src.get("section_type"),
                    "topic": src.get("topic"),
                    "heading": src.get("heading"),
                }
            )
        return hits

    def search(self, query: Any, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search API that accepts either:
          - an embedding (list of floats) -> calls search_with_embedding
          - a query string -> attempts to embed using local BedrockClient (if available), otherwise raises.

        This method is a convenience drop-in for code that calls vs.search(query, top_k).
        """
        # If query already looks like an embedding vector (list/tuple), pass through
        if isinstance(query, (list, tuple)):
            return self.search_with_embedding(list(query), top_k=top_k, filters=filters)

        # If it's a string, try to embed it using BedrockClient if present
        if isinstance(query, str):
            if _HAS_BEDROCK_CLIENT:
                try:
                    bc = BedrockClient(self.config)
                    emb_resp = bc.embed(query)
                    embedding = emb_resp.get("embedding")
                    if not embedding:
                        raise RuntimeError("BedrockClient.embed did not return 'embedding' key")
                    return self.search_with_embedding(embedding, top_k=top_k, filters=filters)
                except Exception as e:
                    print(f"[OpenSearchVectorStore] failed to embed query via BedrockClient: {e}", file=sys.stderr)
                    raise
            else:
                raise RuntimeError(
                    "search() received a string query but no BedrockClient is available to compute embeddings. "
                    "Either pass an embedding list or ensure bedrock_client.py exists and is importable."
                )

        raise ValueError("Unsupported query type for search(); supply a list of floats (embedding) or a query string.")
    
    def delete_index(self):
        """
        Delete the entire index to wipe the knowledge base.
        """
        try:
            self.client.indices.delete(index=self.index_name)
            print(f"[OpenSearchVectorStore] Deleted index '{self.index_name}'", file=sys.stderr)
            # Recreate the index with updated schema
            self._ensure_index()
        except Exception as e:
            print(f"[OpenSearchVectorStore] Error deleting index: {e}", file=sys.stderr)
            raise

