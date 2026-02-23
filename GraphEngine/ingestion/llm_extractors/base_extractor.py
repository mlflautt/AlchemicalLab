"""
LLM-based entity extractors.

Extracts detailed entities using LLM prompts.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import json
import re


@dataclass
class LLMExtractionResult:
    """Result from LLM extraction."""
    characters: List[Dict[str, Any]] = field(default_factory=list)
    locations: List[Dict[str, Any]] = field(default_factory=list)
    factions: List[Dict[str, Any]] = field(default_factory=list)
    items: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)


class BaseLLMExtractor:
    """Base class for LLM-based extraction."""
    
    def __init__(self, llm_client: Callable = None, model: str = "default"):
        self.llm_client = llm_client
        self.model = model
    
    def extract(self, text: str, chunk_id: str = "") -> LLMExtractionResult:
        """Extract entities from text using LLM."""
        if self.llm_client is None:
            return self._mock_extraction(text, chunk_id)
        
        prompt = self._build_extraction_prompt(text)
        response = self.llm_client(prompt)
        
        return self._parse_response(response, chunk_id)
    
    def _build_extraction_prompt(self, text: str) -> str:
        """Build the extraction prompt."""
        return f"""Extract detailed entity information from the following text.

Return a JSON object with these keys:
- characters: List of characters with name, role, description, traits
- locations: List of locations with name, type, description, region
- factions: List of factions with name, ideology, members, goals
- items: List of significant items with name, significance, owner
- events: List of events with name, description, participants
- relationships: List of relationships with source, target, type, context

Text:
{text[:3000]}

JSON:"""
    
    def _parse_response(self, response: str, chunk_id: str) -> LLMExtractionResult:
        """Parse LLM response into structured result."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                return LLMExtractionResult(
                    characters=data.get('characters', []),
                    locations=data.get('locations', []),
                    factions=data.get('factions', []),
                    items=data.get('items', []),
                    events=data.get('events', []),
                    relationships=data.get('relationships', []),
                )
        except json.JSONDecodeError:
            pass
        
        return LLMExtractionResult()
    
    def _mock_extraction(self, text: str, chunk_id: str) -> LLMExtractionResult:
        """Return mock extraction when no LLM available."""
        return LLMExtractionResult()


class CharacterExtractor(BaseLLMExtractor):
    """Extract character information."""
    
    def _build_extraction_prompt(self, text: str) -> str:
        return f"""Extract all characters mentioned in this text.

For each character, provide:
- name: Full name
- role: protagonist, antagonist, mentor, ally, etc.
- description: Physical description if available
- traits: Key personality traits
- motivations: What drives this character
- relationships: How they relate to others

Text:
{text[:2500]}

Return JSON:"""


class LocationExtractor(BaseLLMExtractor):
    """Extract location information."""
    
    def _build_extraction_prompt(self, text: str) -> str:
        return f"""Extract all locations/places mentioned in this text.

For each location, provide:
- name: Name of the place
- type: city, forest, mountain, building, etc.
- description: Physical description
- region: Geographic region if mentioned
- significance: Why this place matters

Text:
{text[:2500]}

Return JSON:"""


class RelationshipExtractor(BaseLLMExtractor):
    """Extract relationships between entities."""
    
    def _build_extraction_prompt(self, text: str) -> str:
        return f"""Extract all relationships between characters/factions mentioned in this text.

For each relationship, provide:
- source: Name of first entity
- target: Name of second entity  
- relationship_type: family, friend, enemy, ally, member, leader, etc.
- context: How this relationship is described
- strength: How strong is this relationship (0.0 to 1.0)

Text:
{text[:2500]}

Return JSON:"""


class NarrativeExtractor(BaseLLMExtractor):
    """Extract narrative arc and thematic information."""
    
    def _build_extraction_prompt(self, text: str) -> str:
        return f"""Analyze this text for narrative elements.

Extract:
- themes: Main themes (good vs evil, redemption, etc.)
- subplots: Secondary storylines
- plot_points: Key moments in the story
- mood: Overall atmosphere

Text:
{text[:2500]}

Return JSON:"""


class LLMExtractorPipeline:
    """
    Pipeline that runs multiple LLM extractors.
    """
    
    def __init__(self, llm_client: Callable = None):
        self.llm_client = llm_client
        
        self.character_extractor = CharacterExtractor(llm_client)
        self.location_extractor = LocationExtractor(llm_client)
        self.relationship_extractor = RelationshipExtractor(llm_client)
        self.narrative_extractor = NarrativeExtractor(llm_client)
    
    def extract_all(self, text: str, chunk_id: str = "") -> LLMExtractionResult:
        """Run all extractors on text."""
        result = LLMExtractionResult()
        
        char_result = self.character_extractor.extract(text, chunk_id)
        result.characters.extend(char_result.characters)
        
        loc_result = self.location_extractor.extract(text, chunk_id)
        result.locations.extend(loc_result.locations)
        
        rel_result = self.relationship_extractor.extract(text, chunk_id)
        result.relationships.extend(rel_result.relationships)
        
        return result
    
    def extract_with_prompt(
        self,
        text: str,
        prompt_type: str = "full"
    ) -> Dict[str, Any]:
        """Extract with custom prompt type."""
        if prompt_type == "characters":
            return self._to_dict(self.character_extractor.extract(text))
        elif prompt_type == "locations":
            return self._to_dict(self.location_extractor.extract(text))
        elif prompt_type == "relationships":
            return self._to_dict(self.relationship_extractor.extract(text))
        elif prompt_type == "narrative":
            return self._to_dict(self.narrative_extractor.extract(text))
        else:
            result = self.extract_all(text)
            return self._to_dict(result)
    
    def _to_dict(self, result: LLMExtractionResult) -> Dict[str, Any]:
        return {
            'characters': result.characters,
            'locations': result.locations,
            'factions': result.factions,
            'items': result.items,
            'events': result.events,
            'relationships': result.relationships,
        }


def create_extractor_pipeline(llm_client: Callable = None) -> LLMExtractorPipeline:
    """Factory function to create extractor pipeline."""
    return LLMExtractorPipeline(llm_client)
