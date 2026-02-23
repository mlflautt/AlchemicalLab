"""
Deterministic extraction using regex patterns.

Fast pattern-based extraction for obvious entities.
"""

from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
import re


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""
    entity_id: str
    entity_type: str
    name: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[int, int] = (0, 0)


@dataclass 
class ExtractedRelationship:
    """A relationship extracted from text."""
    source_id: str
    target_id: str
    relationship_type: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    context: str = ""


class RegexPatterns:
    """Common regex patterns for entity extraction."""
    
    @staticmethod
    def get_dialogue_patterns():
        """Patterns for extracting dialogue."""
        return [
            r'"([^"]+)"\s+said\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'"([^"]+)"\s+says?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'"([^"]+)"\s+replied',
            r'"([^"]+)"\s+answered',
            r'"([^"]+)"\s+whispered',
            r'"([^"]+)"\s+shouted',
        ]
    
    @staticmethod
    def get_name_patterns():
        """Patterns for proper names."""
        return [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',  # First Last
            r'\b([A-Z][a-z]+(?:\s+[A-Z]\.\s+[A-Z][a-z]+))\b',  # First M. Last
        ]
    
    @staticmethod
    def get_date_patterns():
        """Patterns for dates."""
        return [
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
            r'\b(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)\s+(Age|Era|Year)\b',
        ]
    
    @staticmethod
    def get_location_indicators():
        """Keywords that suggest locations."""
        return [
            r'\b(lived in|born in|died in|traveled to|journeyed to|went to|arrived at|departed from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b(in the|at the)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]


class DeterministicExtractor:
    """
    Fast deterministic extraction using regex and patterns.
    
    Extracts:
    - Named entities (via pattern matching)
    - Dialogue
    - Dates/times
    - Location references
    - Chapter/section boundaries
    """
    
    def __init__(self):
        self.patterns = RegexPatterns()
        self.dialogue_re = [re.compile(p, re.IGNORECASE) for p in self.patterns.get_dialogue_patterns()]
        self.name_re = [re.compile(p) for p in self.patterns.get_name_patterns()]
        self.date_re = [re.compile(p, re.IGNORECASE) for p in self.patterns.get_date_patterns()]
        self.location_re = [re.compile(p, re.IGNORECASE) for p in self.patterns.get_location_indicators()]
    
    def extract(self, text: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        Extract entities deterministically from text.
        
        Args:
            text: Text to extract from
            metadata: Optional metadata about the text
        
        Returns:
            Dict with extracted entities and relationships
        """
        result = {
            'dialogue': self.extract_dialogue(text),
            'named_entities': self.extract_named_entities(text),
            'dates': self.extract_dates(text),
            'location_references': self.extract_location_references(text),
        }
        
        return result
    
    def extract_dialogue(self, text: str) -> List[Dict]:
        """Extract dialogue from text."""
        dialogue = []
        
        for pattern in self.dialogue_re:
            matches = pattern.finditer(text)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    dialogue.append({
                        'quote': groups[0].strip(),
                        'speaker': groups[1].strip() if len(groups) > 1 else None,
                        'position': match.span(),
                    })
                elif len(groups) == 1:
                    dialogue.append({
                        'quote': groups[0].strip(),
                        'speaker': None,
                        'position': match.span(),
                    })
        
        return dialogue
    
    def extract_named_entities(self, text: str) -> List[Dict]:
        """Extract proper nouns that might be names."""
        names = set()
        
        for pattern in self.name_re:
            matches = pattern.finditer(text)
            for match in matches:
                name = match.group(1)
                
                excluded_words = {
                    'The Lord', 'The King', 'The Queen', 'Dark Lord',
                    'High King', 'Evil King', 'Good King', 'Old Man',
                    'Young Man', 'Tall Man', 'Short Man', 'Big Man',
                }
                
                if name not in excluded_words:
                    names.add(name)
        
        return [{'name': name, 'type': 'person_guess'} for name in list(names)[:50]]
    
    def extract_dates(self, text: str) -> List[Dict]:
        """Extract date references."""
        dates = []
        
        for pattern in self.date_re:
            matches = pattern.finditer(text)
            for match in matches:
                dates.append({
                    'text': match.group(0),
                    'position': match.span(),
                })
        
        return dates
    
    def extract_location_references(self, text: str) -> List[Dict]:
        """Extract location references."""
        locations = []
        
        for pattern in self.location_re:
            matches = pattern.finditer(text)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    locations.append({
                        'location': groups[1].strip() if len(groups) > 1 else groups[0],
                        'context': groups[0].strip(),
                        'position': match.span(),
                    })
        
        return locations
    
    def extract_chapter_boundaries(self, text: str) -> List[Dict]:
        """Extract chapter/section boundaries."""
        boundaries = []
        
        chapter_patterns = [
            r'^Chapter\s+(\d+|[IVXLC]+)',
            r'^CHAPTER\s+(\d+|[IVXLC]+)',
            r'^Book\s+(\d+|[IVXLC]+)',
            r'^Part\s+(\d+|[IVXLC]+)',
        ]
        
        for pattern in chapter_patterns:
            compiled = re.compile(pattern, re.MULTILINE)
            matches = compiled.finditer(text)
            for match in matches:
                boundaries.append({
                    'type': 'chapter',
                    'title': match.group(0),
                    'position': match.start(),
                })
        
        return boundaries
    
    def get_proper_nouns(self, text: str) -> Set[str]:
        """Extract all capitalized words that might be proper nouns."""
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        excluded = {
            'The', 'A', 'An', 'He', 'She', 'It', 'They', 'We', 'I',
            'When', 'Where', 'What', 'Who', 'How', 'Why', 'Is', 'Are',
            'Was', 'Were', 'Be', 'Been', 'Being', 'Have', 'Has', 'Had',
            'Do', 'Does', 'Did', 'Will', 'Would', 'Could', 'Should', 'May',
            'Might', 'Must', 'Shall', 'Can', 'Need', 'Dare', 'Ought', 'Used',
            'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight',
            'Nine', 'Ten', 'First', 'Second', 'Third', 'Lord', 'King', 'Queen',
        }
        
        proper_nouns = {w for w in words if w not in excluded}
        
        return proper_nouns


def extract_quick_entities(text: str) -> Dict[str, List]:
    """
    Quick entity extraction - simple function for fast results.
    
    Args:
        text: Text to extract from
    
    Returns:
        Dict with simple entity lists
    """
    extractor = DeterministicExtractor()
    
    dialogue = extractor.extract_dialogue(text)
    names = extractor.extract_named_entities(text)
    dates = extractor.extract_dates(text)
    locations = extractor.extract_location_references(text)
    proper_nouns = extractor.get_proper_nouns(text)
    
    return {
        'dialogue': dialogue,
        'names': names,
        'dates': dates,
        'locations': locations,
        'proper_nouns': list(proper_nouns)[:100],
    }
