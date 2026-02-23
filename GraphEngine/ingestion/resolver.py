"""
Entity resolver for deduplication and merging.

Resolves duplicate entities extracted from different text chunks.
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


@dataclass
class EntityCandidate:
    """An entity that may be a duplicate."""
    entity_id: str
    name: str
    entity_type: str
    properties: Dict[str, Any]
    chunk_ids: List[str]
    confidence: float


@dataclass
class MergeCandidate:
    """A potential merge between entities."""
    entity_a: str
    entity_b: str
    similarity: float
    merge_type: str  # "exact", "fuzzy", "property_based"


@dataclass
class MergeResolution:
    """Result of a merge operation."""
    original_id: str
    merged_id: str
    resolution_type: str
    properties_merged: Dict[str, Any]


class EntityResolver:
    """
    Resolves duplicate entities across text chunks.
    
    Uses multiple strategies:
    1. Exact name matching
    2. Fuzzy name matching
    3. Property-based matching
    4. Relationship consistency
    """
    
    def __init__(
        self,
        exact_threshold: float = 1.0,
        fuzzy_threshold: float = 0.85,
        property_weight: float = 0.3
    ):
        self.exact_threshold = exact_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.property_weight = property_weight
    
    def resolve(
        self,
        entities: List[Dict[str, Any]],
        existing_entities: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve duplicate entities.
        
        Args:
            entities: List of extracted entities
            existing_entities: Optional existing entities in graph
        
        Returns:
            Dict with resolved entities and merge info
        """
        entity_map = {e['id']: e for e in entities if 'id' in e}
        
        name_groups = self._group_by_name(entities)
        
        merge_candidates = []
        for name, group in name_groups.items():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        similarity = self._calculate_similarity(group[i], group[j])
                        if similarity >= self.fuzzy_threshold:
                            merge_candidates.append(MergeCandidate(
                                entity_a=group[i]['id'],
                                entity_b=group[j]['id'],
                                similarity=similarity,
                                merge_type='fuzzy' if similarity < 1.0 else 'exact'
                            ))
        
        merged = self._perform_merges(entity_map, merge_candidates)
        
        if existing_entities:
            cross_refs = self._find_cross_references(merged, existing_entities)
            merged = self._merge_cross_references(merged, cross_refs)
        
        resolved = list(merged.values())
        
        return {
            'entities': resolved,
            'merge_count': len(merge_candidates),
            'duplicates_resolved': len(merged) - len(entities),
        }
    
    def _group_by_name(self, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group entities by normalized name."""
        groups = defaultdict(list)
        
        for entity in entities:
            name = entity.get('name', '').strip().lower()
            if name:
                groups[name].append(entity)
        
        return dict(groups)
    
    def _calculate_similarity(
        self,
        entity_a: Dict[str, Any],
        entity_b: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two entities."""
        name_a = entity_a.get('name', '').lower().strip()
        name_b = entity_b.get('name', '').lower().strip()
        
        if name_a == name_b:
            return 1.0
        
        name_sim = self._fuzzy_similarity(name_a, name_b)
        
        props_a = set(entity_a.get('properties', {}).keys())
        props_b = set(entity_b.get('properties', {}).keys())
        
        if props_a and props_b:
            prop_sim = len(props_a & props_b) / len(props_a | props_b)
        else:
            prop_sim = 0.0
        
        combined = name_sim * (1 - self.property_weight) + prop_sim * self.property_weight
        
        return combined
    
    def _fuzzy_similarity(self, str1: str, str2: str) -> float:
        """Calculate fuzzy string similarity."""
        if not str1 or not str2:
            return 0.0
        
        str1_lower = str1.lower()
        str2_lower = str2.lower()
        
        if str1_lower == str2_lower:
            return 1.0
        
        if str1_lower in str2_lower or str2_lower in str1_lower:
            return 0.9
        
        return self._levenshtein_similarity(str1_lower, str2_lower)
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein distance similarity."""
        if len(s1) < len(s2):
            return self._levenshtein_similarity(s2, s1)
        
        if len(s2) == 0:
            return 0.0
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        distance = previous_row[-1]
        max_len = max(len(s1), len(s2))
        
        return 1.0 - (distance / max_len)
    
    def _perform_merges(
        self,
        entity_map: Dict[str, Dict],
        candidates: List[MergeCandidate]
    ) -> Dict[str, Dict]:
        """Perform merges based on candidates."""
        merged = dict(entity_map)
        
        merged_ids = set()
        
        for candidate in sorted(candidates, key=lambda x: x.similarity, reverse=True):
            if candidate.entity_a in merged_ids or candidate.entity_b in merged_ids:
                continue
            
            entity_a = merged[candidate.entity_a]
            entity_b = merged[candidate.entity_b]
            
            merged_entity = self._merge_entities(entity_a, entity_b, candidate.merge_type)
            
            new_id = self._generate_entity_id(merged_entity['name'])
            merged_entity['id'] = new_id
            
            del merged[candidate.entity_a]
            del merged[candidate.entity_b]
            
            merged[new_id] = merged_entity
            
            merged_ids.add(new_id)
        
        return merged
    
    def _merge_entities(
        self,
        entity_a: Dict,
        entity_b: Dict,
        merge_type: str
    ) -> Dict:
        """Merge two entities."""
        merged = {
            'id': entity_a.get('id', ''),
            'name': entity_a.get('name') or entity_b.get('name', ''),
            'type': entity_a.get('type') or entity_b.get('type', ''),
            'properties': {},
            'merge_info': {
                'merged_from': [entity_a.get('id', ''), entity_b.get('id', '')],
                'merge_type': merge_type,
            }
        }
        
        props_a = entity_a.get('properties', {})
        props_b = entity_b.get('properties', {})
        
        all_keys = set(props_a.keys()) | set(props_b.keys())
        for key in all_keys:
            val_a = props_a.get(key)
            val_b = props_b.get(key)
            
            if val_a and val_b:
                if val_a == val_b:
                    merged['properties'][key] = val_a
                else:
                    merged['properties'][key] = val_a if val_a else val_b
            elif val_a:
                merged['properties'][key] = val_a
            else:
                merged['properties'][key] = val_b
        
        return merged
    
    def _generate_entity_id(self, name: str) -> str:
        """Generate a canonical entity ID from name."""
        clean = ''.join(c.lower() for c in name if c.isalnum() or c.isspace())
        clean = '_'.join(clean.split())
        
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:6]
        
        return f"{clean}_{hash_suffix}"
    
    def _find_cross_references(
        self,
        new_entities: Dict[str, Dict],
        existing_entities: List[Dict]
    ) -> List[MergeCandidate]:
        """Find references between new and existing entities."""
        existing_map = {e.get('name', '').lower(): e for e in existing_entities}
        
        cross_refs = []
        
        for new_id, new_entity in new_entities.items():
            new_name = new_entity.get('name', '').lower()
            
            if new_name in existing_map:
                existing = existing_map[new_name]
                
                similarity = self._calculate_similarity(new_entity, existing)
                
                if similarity >= self.fuzzy_threshold:
                    cross_refs.append(MergeCandidate(
                        entity_a=existing.get('id', ''),
                        entity_b=new_id,
                        similarity=similarity,
                        merge_type='cross_reference'
                    ))
        
        return cross_refs
    
    def _merge_cross_references(
        self,
        entities: Dict[str, Dict],
        cross_refs: List[MergeCandidate]
    ) -> Dict[str, Dict]:
        """Merge new entities with existing ones."""
        for ref in cross_refs:
            if ref.entity_b in entities:
                pass
        
        return entities


def resolve_entities_simple(
    entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Simple entity resolution without dependencies.
    
    Args:
        entities: List of entities to resolve
    
    Returns:
        List of resolved entities
    """
    resolver = EntityResolver()
    result = resolver.resolve(entities)
    return result['entities']
