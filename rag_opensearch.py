import streamlit as st
from bedrock_client import BedrockClient
from opensearch_vector_store import OpenSearchVectorStore
from typing import List, Dict, Any
import time

class RAGOpenSearch:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bedrock = BedrockClient(config)
        self.vs = OpenSearchVectorStore(config)
        self.max_rounds = int(config.get('max_rounds', 3))
        self.claude_model = config.get('claude_model')

    def answer(self, query: str, top_k: int =5, rerank_threshold: float=0.65, history: List[Dict[str,str]]=None) -> Dict[str,Any]:
        history = history or []
        round_num = 0
        last_answer = ''
        last_hits = []
        while round_num < self.max_rounds:
            round_num += 1
            hits = self.vs.search(query, top_k=top_k)
            last_hits = hits
            avg_sim = sum(h['score'] for h in hits)/len(hits) if hits else 0.0
            st.session_state['last_meta'] = {'round':round_num, 'num_hits':len(hits), 'avg_score':avg_sim}
            # assemble context
            context = "\n\n".join([f"Source {i+1} ({h.get('source','unknown')}): {h['text'][:2000]}" for i,h in enumerate(hits)])
            prompt = self._build_prompt(query, context, history)
            llm_resp = self.bedrock.generate(prompt=prompt, model=self.claude_model)
            answer = llm_resp.get('output') or llm_resp.get('answer') or str(llm_resp)
            confidence = llm_resp.get('confidence', None)
            last_answer = answer
            if confidence is not None and confidence >= rerank_threshold:
                return {'answer': answer, 'sources': hits, 'meta': st.session_state['last_meta']}
            if avg_sim < rerank_threshold or (confidence is not None and confidence < rerank_threshold):
                # fallback: expand query
                if round_num == 1:
                    expanded = self.bedrock.generate(
                        prompt=("Rewrite the user's query for better document retrieval about the Federal Reserve Beige Books, prioritize community development phrasing:\n\n" + query),
                        model=self.claude_model
                    ).get('output') or query
                    query = expanded
                    top_k = min(top_k*2, 50)
                    continue
                else:
                    top_k = min(top_k*2, 100)
                    continue
            else:
                return {'answer': answer, 'sources': hits, 'meta': st.session_state['last_meta']}
        return {'answer': last_answer, 'sources': last_hits, 'meta': st.session_state.get('last_meta',{})}

    def _build_prompt(self, user_query: str, context_text: str, history: List[Dict[str,str]]) -> str:
        system = (
            "You are a kind, professional assistant specialized in the Federal Reserve Beige Books. "
            "Primary focus: community development. Cite sources when possible, using the provided context."
        )
        convo = "\n".join([f"{h['role']}: {h['content']}" for h in (history or [])])
        prompt = f"{system}\nConversation:\n{convo}\n\nUser question:\n{user_query}\n\nContext:\n{context_text}\n\nAnswer succinctly and professionally. If context is insufficient, say so and ask clarifying questions."
        return prompt
