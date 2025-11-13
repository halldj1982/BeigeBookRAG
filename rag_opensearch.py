import streamlit as st
from bedrock_client import BedrockClient
from opensearch_vector_store import OpenSearchVectorStore
from typing import List, Dict, Any
import time
import re
import json
import sys

class RAGOpenSearch:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bedrock = BedrockClient(config)
        self.vs = OpenSearchVectorStore(config)
        self.max_rounds = int(config.get('max_rounds', 3))
        self.claude_model = config.get('claude_model')
    
    def _extract_yyyymm_from_source(self, source: str) -> str:
        """Extract YYYYMM from filename like 'BeigeBook_20250903.pdf' -> '202509'"""
        match = re.search(r'(\d{8})', source)
        if match:
            return match.group(1)[:6]
        return None
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Use LLM to analyze query and extract metadata"""
        prompt = f"""Analyze this user query about Federal Reserve Beige Books and extract:
1. improved_query: Rewritten query optimized for vector search
2. requested_beigebook: Specific Beige Book edition in YYYYMM format (e.g., "September 2025" → "202509", null if not specified)
3. district: Specific Federal Reserve district (e.g., "Atlanta", "San Francisco", null if not specified)
4. section_type: Type of section ("national_summary", "district_report", null if not specified)

User Query: {query}

Return ONLY a valid JSON object (no extra text). Use null for unspecified fields."""
        
        print(f"[RAG] Analyzing query: {query}", file=sys.stderr)
        
        try:
            response = self.bedrock.generate(prompt=prompt, model=self.claude_model)
            result_text = response.get('output', '{}')
            print(f"[RAG] Query analysis raw response: {result_text[:200]}", file=sys.stderr)
            
            # Extract JSON with nested braces support
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text, re.DOTALL)
            if json_match:
                metadata = json.loads(json_match.group(0))
                print(f"[RAG] Extracted metadata: {metadata}", file=sys.stderr)
                return metadata
        except Exception as e:
            print(f"[RAG] Query analysis failed: {e}", file=sys.stderr)
            st.warning(f"Query analysis failed: {e}")
        
        return {
            'improved_query': query,
            'requested_beigebook': None,
            'district': None,
            'section_type': None
        }
    
    def _filter_chunks_by_metadata(self, chunks: List[Dict], metadata: Dict) -> List[Dict]:
        """Filter chunks based on extracted metadata"""
        filtered = []
        requested_bb = metadata.get('requested_beigebook')
        requested_district = metadata.get('district')
        
        print(f"[RAG] Filtering {len(chunks)} chunks with metadata: bb={requested_bb}, district={requested_district}", file=sys.stderr)
        
        for chunk in chunks:
            # Filter by Beige Book date (YYYYMM from filename)
            if requested_bb:
                chunk_yyyymm = self._extract_yyyymm_from_source(chunk.get('source', ''))
                if not chunk_yyyymm or chunk_yyyymm != requested_bb:
                    print(f"[RAG] Filtered out chunk: source={chunk.get('source')}, extracted_yyyymm={chunk_yyyymm}, requested={requested_bb}", file=sys.stderr)
                    continue
            
            # Filter by district
            if requested_district:
                chunk_district = chunk.get('district')
                if not chunk_district or chunk_district != requested_district:
                    print(f"[RAG] Filtered out chunk: district={chunk_district}, requested={requested_district}", file=sys.stderr)
                    continue
            
            filtered.append(chunk)
        
        print(f"[RAG] After filtering: {len(filtered)} chunks remaining", file=sys.stderr)
        
        # Log sample of filtered chunks
        for i, chunk in enumerate(filtered[:3]):
            print(f"[RAG] Kept chunk {i+1}: source={chunk.get('source')}, district={chunk.get('district')}, section={chunk.get('section_type')}", file=sys.stderr)
        
        return filtered
    
    def _score_relevance(self, original_query: str, improved_query: str, chunks: List[Dict], metadata: Dict) -> Dict:
        """Use LLM to score chunk relevance"""
        if not chunks:
            return {'overall_confidence': 0.0, 'chunk_scores': [], 'recommendation': 'insufficient'}
        
        chunk_texts = []
        for i, chunk in enumerate(chunks[:10]):
            chunk_texts.append(
                f"[{i+1}] Source: {chunk.get('source', 'unknown')}, District: {chunk.get('district', 'N/A')}, "
                f"Section: {chunk.get('section_type', 'N/A')}\nText: {chunk.get('text', '')[:500]}"
            )
        
        prompt = f"""Evaluate document relevance for a RAG system.

Original Query: {original_query}
Improved Query: {improved_query}

User Intent:
- Requested Beige Book: {metadata.get('requested_beigebook') or 'any'}
- Requested District: {metadata.get('district') or 'any'}
- Preferred Section Type: {metadata.get('section_type') or 'any'}

Retrieved Chunks:
{chr(10).join(chunk_texts)}

Evaluate each chunk:
1. Relevance to query
2. If user specified section_type, heavily weight matching chunks (+0.2), penalize nulls (-0.1)
3. Information quality

Return ONLY a valid JSON object with exactly this format (no extra text):
{{
  "overall_confidence": 0.75,
  "recommendation": "sufficient"
}}

Valid recommendation values: sufficient, expand_search, insufficient"""
        
        try:
            response = self.bedrock.generate(prompt=prompt, model=self.claude_model)
            result_text = response.get('output', '{}')
            print(f"[RAG] Relevance scoring raw response: {result_text[:200]}", file=sys.stderr)
            
            # Try multiple JSON extraction methods
            parsed = None
            
            # Method 1: Find JSON object with nested braces
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                except:
                    pass
            
            # Method 2: Extract just the values if JSON parsing fails
            if not parsed:
                confidence_match = re.search(r'"overall_confidence"\s*:\s*([0-9.]+)', result_text)
                recommendation_match = re.search(r'"recommendation"\s*:\s*"(\w+)"', result_text)
                if confidence_match:
                    parsed = {
                        'overall_confidence': float(confidence_match.group(1)),
                        'recommendation': recommendation_match.group(1) if recommendation_match else 'sufficient'
                    }
            
            if parsed:
                print(f"[RAG] Parsed relevance score: confidence={parsed.get('overall_confidence')}, recommendation={parsed.get('recommendation')}", file=sys.stderr)
                return parsed
                
        except Exception as e:
            print(f"[RAG] Relevance scoring failed: {e}", file=sys.stderr)
            st.warning(f"Relevance scoring failed: {e}")
        
        # Fallback: assume medium confidence
        print(f"[RAG] Using fallback relevance score", file=sys.stderr)
        return {'overall_confidence': 0.5, 'recommendation': 'sufficient'}

    def answer(self, query: str, top_k: int =5, rerank_threshold: float=0.65, history: List[Dict[str,str]]=None) -> Dict[str,Any]:
        history = history or []
        original_query = query
        
        # Analyze query and extract metadata
        query_metadata = self._analyze_query(query)
        improved_query = query_metadata.get('improved_query', query)
        
        round_num = 0
        last_answer = ''
        last_hits = []
        final_confidence = 0.0
        
        while round_num < self.max_rounds:
            round_num += 1
            
            # Vector search with improved query
            print(f"[RAG] Round {round_num}: Searching with query='{improved_query}', top_k={top_k}", file=sys.stderr)
            hits = self.vs.search(improved_query, top_k=top_k)
            print(f"[RAG] Vector search returned {len(hits)} hits", file=sys.stderr)
            
            # Filter by metadata
            filtered_hits = self._filter_chunks_by_metadata(hits, query_metadata)
            
            # LLM-based relevance scoring
            relevance_result = self._score_relevance(original_query, improved_query, filtered_hits, query_metadata)
            confidence = relevance_result.get('overall_confidence', 0.0)
            final_confidence = confidence
            
            last_hits = filtered_hits
            st.session_state['last_meta'] = {
                'round': round_num,
                'num_hits': len(hits),
                'filtered_hits': len(filtered_hits),
                'confidence': confidence,
                'metadata': query_metadata
            }
            
            # Assemble context
            context_parts = []
            for i, h in enumerate(filtered_hits[:10]):
                meta_info = []
                if h.get('district'):
                    meta_info.append(f"District: {h['district']}")
                if h.get('section_type'):
                    meta_info.append(f"Section: {h['section_type']}")
                meta_str = " | ".join(meta_info) if meta_info else h.get('source', 'unknown')
                context_parts.append(f"[{i+1}] {meta_str}\n{h['text'][:2000]}")
            context = "\n\n".join(context_parts)
            
            # Generate answer with confidence level
            prompt = self._build_prompt(original_query, context, history, confidence)
            llm_resp = self.bedrock.generate(prompt=prompt, model=self.claude_model)
            answer = llm_resp.get('output') or llm_resp.get('answer') or str(llm_resp)
            last_answer = answer
            
            # Check if confidence meets threshold
            if confidence >= rerank_threshold:
                return {'answer': answer, 'sources': filtered_hits, 'meta': st.session_state['last_meta']}
            
            # Expand search if confidence low
            if round_num < self.max_rounds:
                top_k = min(top_k * 2, 100)
                continue
        
        # Final answer with low confidence indicator
        return {'answer': last_answer, 'sources': last_hits, 'meta': st.session_state.get('last_meta', {})}

    def _build_prompt(self, user_query: str, context_text: str, history: List[Dict[str,str]], confidence: float = 1.0) -> str:
        confidence_level = "HIGH" if confidence >= 0.75 else "MEDIUM" if confidence >= 0.5 else "LOW"
        
        low_confidence_instruction = ""
        if confidence < 0.65:
            low_confidence_instruction = (
                "\n\nCONFIDENCE LEVEL: LOW\n"
                "The available context has limited information for this query. "
                "Begin your response by acknowledging this limitation using varied phrasing such as:\n"
                "- 'Based on the available Beige Book data, information on this topic is limited...'\n"
                "- 'My knowledge base contains sparse information about this specific question...'\n"
                "- 'The Beige Books I have access to provide only partial insight into...'\n"
                "Then provide the best possible answer from the available context."
            )
        
        system = (
            "You are BeigeBot, a professional assistant specialized in the Federal Reserve Beige Books. "
            "Your purpose is to help users understand economic conditions and trends across Federal Reserve districts.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Use the provided context sources which include metadata (District, Section)\n"
            "- When citing information, use numbered references like [1], [2], etc.\n"
            "- At the end of your response, include a 'References:' section listing each source\n"
            "- Format references as: [1] District Name - Section Type\n"
            "- Be concise and professional."
            f"{low_confidence_instruction}"
        )
        convo = "\n".join([f"{h['role']}: {h['content']}" for h in (history or [])])
        prompt = f"{system}\n\nConversation:\n{convo}\n\nUser question:\n{user_query}\n\nContext Sources:\n{context_text}\n\nProvide your answer with numbered citations and a References section at the end."
        return prompt
