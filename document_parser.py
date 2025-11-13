import re
from typing import List, Dict, Any
from datetime import datetime

class BeigeBookParser:
    DISTRICTS = {
        "Boston": 1, "New York": 2, "Philadelphia": 3, "Cleveland": 4,
        "Richmond": 5, "Atlanta": 6, "Chicago": 7, "St. Louis": 8,
        "Minneapolis": 9, "Kansas City": 10, "Dallas": 11, "San Francisco": 12
    }
    
    TOPICS = [
        "Overall Economic Activity", "Labor Markets", "Prices", "Consumer Spending",
        "Manufacturing", "Real Estate", "Financial Services", "Agriculture",
        "Energy", "Transportation", "Nonfinancial Services", "Employment",
        "Wages", "Construction"
    ]
    
    def __init__(self, text: str, source_name: str = None):
        self.text = text
        self.source_name = source_name or "beige-book"
        self.publication_date = self._extract_date()
        
    def _extract_date(self) -> str:
        match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', self.text[:500])
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return "Unknown"
    
    def _detect_section_type(self, text: str) -> str:
        if "National Summary" in text[:100]:
            return "national_summary"
        elif "About This Publication" in text[:100]:
            return "about"
        for district in self.DISTRICTS.keys():
            if f"Federal Reserve Bank of {district}" in text[:100]:
                return "district_report"
        return "other"
    
    def _detect_district(self, text: str) -> tuple:
        for district, number in self.DISTRICTS.items():
            if f"Federal Reserve Bank of {district}" in text[:200]:
                return district, number
        return None, None
    
    def _detect_topic(self, text: str) -> str:
        for topic in self.TOPICS:
            if topic in text[:500]:
                return topic
        return None
    
    def _split_into_sections(self) -> List[Dict[str, Any]]:
        sections = []
        lines = self.text.split('\n')
        current_section = []
        current_heading = None
        skip_section = False
        
        for line in lines:
            stripped = line.strip()
            
            # Detect major section breaks
            is_district_header = any(f"Federal Reserve Bank of {d}" in stripped for d in self.DISTRICTS.keys())
            is_national_summary = "National Summary" in stripped
            is_about = "About This Publication" in stripped
            is_contents = "Contents" in stripped and len(stripped) < 20
            
            if is_district_header or is_national_summary or is_about or is_contents:
                # Save previous section if not skipped
                if current_section and not skip_section:
                    sections.append({
                        'heading': current_heading,
                        'text': '\n'.join(current_section)
                    })
                
                # Start new section
                current_section = [line]
                current_heading = stripped
                
                # Mark low-value sections to skip
                skip_section = is_about or is_contents
            else:
                current_section.append(line)
        
        # Save final section if not skipped
        if current_section and not skip_section:
            sections.append({
                'heading': current_heading,
                'text': '\n'.join(current_section)
            })
        
        return sections
    
    def _chunk_with_overlap(self, text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        
        while i < len(words):
            end = min(i + chunk_size, len(words))
            chunk = ' '.join(words[i:end])
            chunks.append(chunk)
            i += chunk_size - overlap
            
            if end >= len(words):
                break
        
        return chunks
    
    def parse(self, chunk_size: int = 600) -> List[Dict[str, Any]]:
        sections = self._split_into_sections()
        parsed_chunks = []
        chunk_index = 0
        
        for section in sections:
            section_text = section['text']
            heading = section['heading']
            
            section_type = self._detect_section_type(section_text)
            district, district_number = self._detect_district(section_text)
            
            # Split section into paragraphs
            paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
            
            # Combine paragraphs into chunks
            current_chunk = []
            current_words = 0
            current_topic = self._detect_topic(heading or '')
            
            for para in paragraphs:
                para_words = len(para.split())
                
                # Detect topic from paragraph if not already set
                if not current_topic:
                    current_topic = self._detect_topic(para)
                
                if current_words + para_words > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    if heading:
                        chunk_text = f"{heading}\n\n{chunk_text}"
                    
                    parsed_chunks.append({
                        'text': chunk_text,
                        'chunk_index': chunk_index,
                        'section_type': section_type,
                        'district': district,
                        'district_number': district_number,
                        'topic': current_topic,
                        'heading': heading,
                        'word_count': len(chunk_text.split()),
                        'publication_date': self.publication_date,
                        'source': self.source_name
                    })
                    chunk_index += 1
                    current_chunk = [para]
                    current_words = para_words
                    current_topic = self._detect_topic(para)
                else:
                    current_chunk.append(para)
                    current_words += para_words
            
            # Save remaining chunk
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                if heading:
                    chunk_text = f"{heading}\n\n{chunk_text}"
                
                parsed_chunks.append({
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'section_type': section_type,
                    'district': district,
                    'district_number': district_number,
                    'topic': current_topic,
                    'heading': heading,
                    'word_count': len(chunk_text.split()),
                    'publication_date': self.publication_date,
                    'source': self.source_name
                })
                chunk_index += 1
        
        return parsed_chunks
