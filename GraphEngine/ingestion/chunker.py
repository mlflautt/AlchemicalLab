"""
Text chunking strategies for ingestion.

Breaks text into manageable chunks for processing.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import os


@dataclass
class TextChunk:
    """A chunk of text for processing."""
    chunk_id: str
    content: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass
class IngestOptions:
    """Options for ingestion."""
    source_type: str = "novel"
    chunk_by: str = "auto"  # "chapters", "scenes", "size", "auto"
    max_chunk_tokens: int = 2000
    chunk_overlap: int = 200
    extract_deterministic: bool = True
    extract_with_llm: bool = True
    validate: bool = True
    analyze_narrative: bool = True


class TextChunker:
    """
    Chunks text into manageable sections for processing.
    
    Supports multiple chunking strategies:
    - auto: Detect source type and choose appropriate strategy
    - chapters: Chunk by chapter headings (novels)
    - scenes: Chunk by scene breaks (scripts)
    - size: Fixed-size chunks with overlap
    """
    
    CHAPTER_PATTERNS = [
        r'^Chapter\s+(\d+|[IVXLC]+)',
        r'^CHAPTER\s+(\d+|[IVXLC]+)',
        r'^Book\s+(\d+|[IVXLC]+)',
        r'^BOOK\s+(\d+|[IVXLC]+)',
        r'^Part\s+(\d+|[IVXLC]+)',
        r'^PART\s+(\d+|[IVXLC]+)',
        r'^\d+\.\s+[A-Z]',  # Numbered sections
    ]
    
    SCENE_PATTERNS = [
        r'^INT\.',
        r'^EXT\.',
        r'^INT/EXT',
        r'^FADE IN:',
        r'^\s*\[.*\]\s*$',  # Stage directions
    ]
    
    def __init__(self):
        self.chapter_re = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.CHAPTER_PATTERNS]
        self.scene_re = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.SCENE_PATTERNS]
    
    def chunk(
        self,
        text: str,
        source_type: str = "novel",
        chunk_by: str = "auto",
        max_tokens: int = 2000,
        overlap: int = 200
    ) -> List[TextChunk]:
        """
        Chunk text based on strategy.
        
        Args:
            text: Text to chunk
            source_type: Type of source ("novel", "script", "prompt")
            chunk_by: Chunking strategy
            max_tokens: Maximum tokens per chunk
            overlap: Overlap between chunks in words
        
        Returns:
            List of TextChunk objects
        """
        if chunk_by == "auto":
            chunk_by = self._detect_chunk_strategy(source_type, text)
        
        if chunk_by == "chapters":
            return self.chunk_by_chapters(text)
        elif chunk_by == "scenes":
            return self.chunk_by_scenes(text)
        elif chunk_by == "size":
            return self.chunk_by_size(text, max_tokens, overlap)
        else:
            return self.chunk_by_size(text, max_tokens, overlap)
    
    def _detect_chunk_strategy(self, source_type: str, text: str) -> str:
        """Detect appropriate chunking strategy."""
        if source_type == "script":
            if self._has_scene_markers(text):
                return "scenes"
            return "size"
        elif source_type == "novel":
            if self._has_chapter_markers(text):
                return "chapters"
            return "size"
        else:
            return "size"
    
    def _has_chapter_markers(self, text: str) -> bool:
        """Check if text has chapter markers."""
        for pattern in self.chapter_re:
            if pattern.search(text):
                return True
        return False
    
    def _has_scene_markers(self, text: str) -> bool:
        """Check if text has scene markers."""
        for pattern in self.scene_re:
            if pattern.search(text):
                return True
        return False
    
    def chunk_by_chapters(self, text: str) -> List[TextChunk]:
        """Chunk text by chapter headings."""
        chunks = []
        lines = text.split('\n')
        
        current_chapter = None
        current_content = []
        current_start = 0
        chunk_counter = 0
        
        for i, line in enumerate(lines):
            is_chapter = False
            chapter_num = None
            
            for pattern in self.chapter_re:
                match = pattern.match(line.strip())
                if match:
                    is_chapter = True
                    chapter_num = match.group(1) if match.lastindex else None
                    break
            
            if is_chapter and current_chapter is not None:
                content = '\n'.join(current_content)
                if content.strip():
                    chunks.append(TextChunk(
                        chunk_id=f"chapter_{chunk_counter}",
                        content=content.strip(),
                        start_index=current_start,
                        end_index=i,
                        metadata={"chapter": current_chapter}
                    ))
                    chunk_counter += 1
                
                current_chapter = chapter_num or str(chunk_counter + 1)
                current_content = [line]
                current_start = i
            else:
                current_content.append(line)
                if current_chapter is None:
                    current_chapter = "Prologue"
        
        if current_content:
            content = '\n'.join(current_content)
            if content.strip():
                chunks.append(TextChunk(
                    chunk_id=f"chapter_{chunk_counter}",
                    content=content.strip(),
                    start_index=current_start,
                    end_index=len(text),
                    metadata={"chapter": current_chapter}
                ))
        
        return chunks
    
    def chunk_by_scenes(self, text: str) -> List[TextChunk]:
        """Chunk text by scene markers (scripts)."""
        chunks = []
        lines = text.split('\n')
        
        current_scene = []
        current_start = 0
        scene_counter = 0
        
        for i, line in enumerate(lines):
            is_scene = False
            
            for pattern in self.scene_re:
                if pattern.match(line.strip()):
                    is_scene = True
                    break
            
            if is_scene and current_scene:
                content = '\n'.join(current_scene)
                if content.strip():
                    chunks.append(TextChunk(
                        chunk_id=f"scene_{scene_counter}",
                        content=content.strip(),
                        start_index=current_start,
                        end_index=i,
                        metadata={"scene_number": scene_counter + 1}
                    ))
                    scene_counter += 1
                
                current_scene = [line]
                current_start = i
            else:
                current_scene.append(line)
        
        if current_scene:
            content = '\n'.join(current_scene)
            if content.strip():
                chunks.append(TextChunk(
                    chunk_id=f"scene_{scene_counter}",
                    content=content.strip(),
                    start_index=current_start,
                    end_index=len(text),
                    metadata={"scene_number": scene_counter + 1}
                ))
        
        return chunks
    
    def chunk_by_size(
        self,
        text: str,
        max_tokens: int = 2000,
        overlap: int = 200
    ) -> List[TextChunk]:
        """Chunk text by size with overlap."""
        words = text.split()
        chunk_size = max_tokens * 3 // 4  # Rough token estimate
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            content = ' '.join(chunk_words)
            
            chunks.append(TextChunk(
                chunk_id=f"chunk_{chunk_id}",
                content=content,
                start_index=start,
                end_index=end,
                metadata={"chunk_number": chunk_id + 1}
            ))
            
            chunk_id += 1
            start = end - overlap
            
            if start >= len(words):
                break
        
        return chunks
    
    def detect_chapters(self, text: str) -> List[Dict[str, Any]]:
        """Get chapter metadata without full chunking."""
        chapters = []
        lines = text.split('\n')
        
        current_chapter_start = 0
        current_chapter = None
        
        for i, line in enumerate(lines):
            for pattern in self.chapter_re:
                match = pattern.match(line.strip())
                if match:
                    chapter_num = match.group(1) if match.lastindex else str(len(chapters) + 1)
                    
                    if current_chapter:
                        chapters.append({
                            "number": current_chapter,
                            "start": current_chapter_start,
                            "end": i
                        })
                    
                    current_chapter = chapter_num
                    current_chapter_start = i
                    break
        
        if current_chapter:
            chapters.append({
                "number": current_chapter,
                "start": current_chapter_start,
                "end": len(lines)
            })
        
        return chapters
    
    def get_word_count(self, text: str) -> int:
        """Get approximate word count."""
        return len(text.split())
    
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count (rough)."""
        word_count = self.get_word_count(text)
        return word_count * 4 // 3


def load_text_file(filepath: str) -> Tuple[str, Dict[str, Any]]:
    """
    Load text from file with metadata.
    
    Args:
        filepath: Path to text file
    
    Returns:
        Tuple of (text, metadata)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    metadata = {
        "filename": os.path.basename(filepath),
        "size_bytes": os.path.getsize(filepath),
    }
    
    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    
    elif ext in ['.md', '.markdown']:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        metadata['type'] = 'markdown'
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    metadata['word_count'] = len(text.split())
    
    return text, metadata


def detect_source_type(text: str, filename: str = None) -> str:
    """Detect source type from text content."""
    has_scenes = bool(re.search(r'^(INT|EXT)\.', text, re.MULTILINE | re.IGNORECASE))
    if has_scenes:
        return "script"
    
    has_chapters = bool(re.search(r'^Chapter\s+\d+', text, re.MULTILINE | re.IGNORECASE))
    if has_chapters:
        return "novel"
    
    if filename:
        lower = filename.lower()
        if 'script' in lower or 'screenplay' in lower:
            return "script"
        if 'novel' in lower or 'book' in lower:
            return "novel"
    
    return "prompt"
