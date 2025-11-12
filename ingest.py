import os, uuid, math, sys
from typing import Dict, Any
from pdfminer.high_level import extract_text
from bedrock_client import BedrockClient
from opensearch_vector_store import OpenSearchVectorStore

class Ingestor:
    def __init__(self, config: Dict[str,Any]):
        self.config = config
        self.bedrock = BedrockClient(config)
        self.vs = OpenSearchVectorStore(config)
        self.chunk_size = int(config.get('chunk_size', 800))

    def _chunk_text(self, text: str):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i+self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size
        return chunks

    def ingest_pdf(self, path: str, source_name: str = None) -> Dict[str,Any]:
        text = extract_text(path)
        # naive split by pages or chunk
        chunks = self._chunk_text(text)
        num_docs = 0
        for idx, c in enumerate(chunks):
            emb_resp = self.bedrock.embed(c)
            embedding = emb_resp.get('embedding')
            if not embedding or not isinstance(embedding, list) or len(embedding) == 0:
                continue
                
            doc_id = str(uuid.uuid4())
            self.vs.index_document(doc_id=doc_id, text=c, embedding=embedding, source=source_name, page=idx+1)
            num_docs += 1
        return {'num_docs': num_docs, 'num_pages': len(chunks)}

