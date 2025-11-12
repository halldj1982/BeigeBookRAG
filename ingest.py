import os, uuid, math, sys
from typing import Dict, Any
from pdfminer.high_level import extract_text
from bedrock_client import BedrockClient
from opensearch_vector_store import OpenSearchVectorStore
from document_parser import BeigeBookParser

class Ingestor:
    def __init__(self, config: Dict[str,Any]):
        self.config = config
        self.bedrock = BedrockClient(config)
        self.vs = OpenSearchVectorStore(config)
        self.chunk_size = int(config.get('chunk_size', 600))

    def ingest_pdf(self, path: str, source_name: str = None) -> Dict[str,Any]:
        text = extract_text(path)
        return self.ingest_text(text, source_name)
    
    def ingest_text(self, text: str, source_name: str = None) -> Dict[str,Any]:
        parser = BeigeBookParser(text, source_name)
        chunks = parser.parse(chunk_size=self.chunk_size)
        
        num_docs = 0
        for chunk_data in chunks:
            emb_resp = self.bedrock.embed(chunk_data['text'])
            embedding = emb_resp.get('embedding')
            if not embedding or not isinstance(embedding, list) or len(embedding) == 0:
                continue
            
            doc_id = str(uuid.uuid4())
            self.vs.index_document(doc_id=doc_id, text=chunk_data['text'], embedding=embedding, metadata=chunk_data)
            num_docs += 1
        
        return {'num_docs': num_docs, 'num_chunks': len(chunks)}

