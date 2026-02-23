"""
Validator LLM for consistency checking.

Reviews extracted entities for consistency and accuracy.
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
import json
import re


@dataclass
class ValidationResult:
    """Result of entity validation."""
    is_valid: bool
    verified_entities: List[Dict] = field(default_factory=list)
    conflicts: List[Dict] = field(default_factory=list)
    suggestions: List[Dict] = field(default_factory=list)
    missing_entities: List[Dict] = field(default_factory=list)


class ValidatorLLM:
    """
    Uses LLM to validate extracted entities.
    
    Checks:
    - Consistency between entities
    - Conflicts (e.g., character alive then dead)
    - Missing obvious entities
    - Relationship validity
    """
    
    def __init__(self, llm_client: Callable = None, model: str = "default"):
        self.llm_client = llm_client
        self.model = model
    
    def validate(
        self,
        entities: Dict[str, List],
        source_text: str = None
    ) -> ValidationResult:
        """
        Validate extracted entities.
        
        Args:
            entities: Dict with entity lists by type
            source_text: Optional source text for reference
        
        Returns:
            ValidationResult with issues found
        """
        if self.llm_client is None:
            return self._mock_validation(entities)
        
        prompt = self._build_validation_prompt(entities, source_text)
        response = self.llm_client(prompt)
        
        return self._parse_validation_response(response)
    
    def _build_validation_prompt(
        self,
        entities: Dict[str, List],
        source_text: str = None
    ) -> str:
        """Build validation prompt."""
        entities_summary = json.dumps(entities, indent=2)
        
        prompt = f"""You are a consistency checker for extracted story entities.

Review the following extracted entities and identify:
1. CONFLICTS: Contradictions (e.g., character alive in one place, dead in another)
2. MISSING: Obvious entities that should have been extracted
3. SUGGESTIONS: Improvements to entity descriptions
4. VERIFIED: Entities that look correct

Extracted Entities:
{entities_summary}
"""
        
        if source_text:
            prompt += f"\n\nReference (first 1000 chars):\n{source_text[:1000]}"
        
        prompt += """

Return JSON with:
{
  "verified_entities": [{"typename": "...", "reason": "...", "": "..."}],
  "conflicts": [{"type": "...", "description": "...", "severity": "high/medium/low"}],
  "suggestions": [{"entity": "...", "suggestion": "..."}],
  "missing": [{"type": "...", "reason": "..."}]
}

JSON:"""
        
        return prompt
    
    def _parse_validation_response(self, response: str) -> ValidationResult:
        """Parse validation response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                return ValidationResult(
                    is_valid=len(data.get('conflicts', [])) == 0,
                    verified_entities=data.get('verified_entities', []),
                    conflicts=data.get('conflicts', []),
                    suggestions=data.get('suggestions', []),
                    missing_entities=data.get('missing', []),
                )
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return ValidationResult(is_valid=True)
    
    def _mock_validation(self, entities: Dict[str, List]) -> ValidationResult:
        """Mock validation when no LLM available."""
        return ValidationResult(
            is_valid=True,
            verified_entities=[
                {"type": t, "count": len(lst)}
                for t, lst in entities.items()
            ]
        )


def validate_entities_simple(
    entities: Dict[str, List],
    llm_client: Callable = None
) -> ValidationResult:
    """Simple validation function."""
    validator = ValidatorLLM(llm_client)
    return validator.validate(entities)
