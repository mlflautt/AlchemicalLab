"""
Context Extractor for GraphEngine.

Provides deterministic, budget-aware context extraction from the knowledge graph.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
import heapq

from GraphEngine.context.templates import (
    TaskType, TaskTemplate, ContextRequirement,
    get_template, get_requirements, estimate_tokens
)
from GraphEngine.context.serializer import PromptSerializer, truncate_to_budget


@dataclass
class ScoredNode:
    """Node with relevance score."""
    node: Any
    score: float
    distance: int
    
    def __lt__(self, other):
        return self.score > other.score


@dataclass
class ExtractedContext:
    """Extracted context package."""
    focus_entity: Any
    related_entities: List[Any]
    relationships: List[Dict]
    recent_events: List[Any]
    active_conflicts: List[Dict]
    temporal_context: Dict[str, Any]
    world_state: Dict[str, Any]
    token_count: int
    extraction_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'focus_entity': self.focus_entity,
            'related_entities': self.related_entities,
            'relationships': self.relationships,
            'recent_events': self.recent_events,
            'active_conflicts': self.active_conflicts,
            'temporal_context': self.temporal_context,
            'world_state': self.world_state,
            'token_count': self.token_count,
            'extraction_metadata': self.extraction_metadata,
        }
    
    def serialize(self, format: str = 'markdown') -> str:
        """Serialize context to prompt format."""
        return PromptSerializer.serialize(self.to_dict(), format, self.extraction_metadata.get('task_type'))


class ContextExtractor:
    """
    Extract deterministic, budget-aware context from knowledge graph.
    
    Features:
    - Task-specific extraction templates
    - Importance-weighted node selection
    - Token budget awareness
    - Temporal context inclusion
    - Conflict detection
    """
    
    IMPORTANCE_WEIGHTS = {
        'edge_strength': 0.30,
        'node_recency': 0.20,
        'node_importance': 0.20,
        'relationship_type': 0.15,
        'distance': 0.15,
    }
    
    RELATIONSHIP_TYPE_PRIORITY = {
        'conflict': 1.0,
        'alliance': 0.9,
        'predation': 0.8,
        'competition': 0.7,
        'mutualism': 0.6,
        'dependency': 0.5,
        'contains': 0.4,
        'origin': 0.4,
        'references': 0.2,
        'similar_to': 0.1,
    }
    
    def __init__(self, graph: 'KnowledgeGraph'):
        self.graph = graph
    
    def get_context(
        self,
        entity_id: str,
        task_type: str = 'scene_generation',
        max_tokens: int = None,
        additional_requirements: Dict[str, Any] = None
    ) -> ExtractedContext:
        """
        Extract context for an entity and task.
        
        Args:
            entity_id: ID of the focus entity
            task_type: Type of generation task
            max_tokens: Maximum tokens for context (uses template default if None)
            additional_requirements: Override template requirements
        
        Returns:
            ExtractedContext with all relevant information
        """
        task_type_enum = TaskType(task_type) if isinstance(task_type, str) else task_type
        template = get_template(task_type_enum)
        
        if not template:
            template = get_template(TaskType.SCENE_GENERATION)
        
        requirements = template.requirements
        if additional_requirements:
            requirements = ContextRequirement(
                node_types=additional_requirements.get('node_types', requirements.node_types),
                edge_types=additional_requirements.get('edge_types', requirements.edge_types),
                max_depth=additional_requirements.get('max_depth', requirements.max_depth),
                max_nodes=additional_requirements.get('max_nodes', requirements.max_nodes),
                importance_threshold=additional_requirements.get('importance_threshold', requirements.importance_threshold),
                include_temporal=additional_requirements.get('include_temporal', requirements.include_temporal),
                include_conflicts=additional_requirements.get('include_conflicts', requirements.include_conflicts),
            )
        
        max_tokens = max_tokens or template.max_tokens
        
        focus_entity = self.graph.get_node(entity_id)
        if not focus_entity:
            raise ValueError(f"Entity not found: {entity_id}")
        
        if focus_entity.type not in template.focus_entity_types:
            pass
        
        scored_nodes = self._extract_related_nodes(
            entity_id=entity_id,
            requirements=requirements
        )
        
        related_entities = self._select_top_nodes(scored_nodes, requirements.max_nodes)
        
        relationships = self._extract_relationships(
            entity_id=entity_id,
            related_ids=[n.id for n in related_entities],
            edge_types=requirements.edge_types
        )
        
        recent_events = []
        if requirements.include_temporal:
            recent_events = self._extract_recent_events(
                entity_id=entity_id,
                related_ids=[n.id for n in related_entities],
                limit=10
            )
        
        active_conflicts = []
        if requirements.include_conflicts:
            active_conflicts = self._extract_conflicts(
                entity_id=entity_id,
                related_ids=[n.id for n in related_entities]
            )
        
        temporal_context = {}
        if requirements.include_temporal:
            temporal_context = self._get_temporal_context(focus_entity)
        
        world_state = self._get_world_state_summary()
        
        context = ExtractedContext(
            focus_entity=focus_entity,
            related_entities=related_entities,
            relationships=relationships,
            recent_events=recent_events,
            active_conflicts=active_conflicts,
            temporal_context=temporal_context,
            world_state=world_state,
            token_count=0,
            extraction_metadata={
                'task_type': task_type_enum.value,
                'focus_id': entity_id,
                'focus_type': focus_entity.type,
                'extraction_time': datetime.utcnow().isoformat(),
                'template_used': template.task_type.value,
                'requirements': {
                    'max_nodes': requirements.max_nodes,
                    'max_depth': requirements.max_depth,
                    'node_types': requirements.node_types,
                },
            }
        )
        
        context.token_count = self._estimate_tokens(context)
        
        if context.token_count > max_tokens:
            context = self._prune_to_budget(context, max_tokens)
        
        return context
    
    def get_context_for_characters(
        self,
        character_ids: List[str],
        task_type: str = 'scene_generation',
        max_tokens: int = None
    ) -> ExtractedContext:
        """
        Extract context involving multiple characters (e.g., for dialogue).
        
        Args:
            character_ids: List of character IDs
            task_type: Type of generation task
            max_tokens: Maximum tokens
        
        Returns:
            ExtractedContext with all characters and their shared context
        """
        if not character_ids:
            raise ValueError("No character IDs provided")
        
        task_type_enum = TaskType(task_type) if isinstance(task_type, str) else task_type
        template = get_template(task_type_enum)
        max_tokens = max_tokens or (template.max_tokens if template else 2000)
        
        characters = [self.graph.get_node(cid) for cid in character_ids]
        characters = [c for c in characters if c]
        
        if not characters:
            raise ValueError("No valid characters found")
        
        focus_entity = characters[0]
        
        all_related = set()
        for char in characters:
            neighbors = self.graph.get_neighbors(char.id, depth=1)
            all_related.update(neighbors)
        
        related_entities = []
        for node_id in all_related:
            node = self.graph.get_node(node_id)
            if node and node.id not in character_ids:
                related_entities.append(node)
        
        relationships = []
        for char in characters:
            edges = self.graph.get_edges(char.id)
            for edge in edges:
                rel = {
                    'source_id': edge.source_id,
                    'target_id': edge.target_id,
                    'edge_type': edge.edge_type,
                    'weight': edge.weight,
                    'context': edge.context,
                }
                source = self.graph.get_node(edge.source_id)
                target = self.graph.get_node(edge.target_id)
                if source:
                    rel['source_name'] = source.properties.get('name', source.id)
                if target:
                    rel['target_name'] = target.properties.get('name', target.id)
                relationships.append(rel)
        
        recent_events = self._extract_recent_events(
            entity_id=characters[0].id,
            related_ids=list(all_related),
            limit=5
        )
        
        active_conflicts = []
        for char in characters:
            conflicts = self._extract_conflicts(char.id, list(all_related))
            active_conflicts.extend(conflicts)
        
        world_state = self._get_world_state_summary()
        
        context = ExtractedContext(
            focus_entity=focus_entity,
            related_entities=related_entities[:15],
            relationships=relationships[:20],
            recent_events=recent_events,
            active_conflicts=active_conflicts[:5],
            temporal_context={},
            world_state=world_state,
            token_count=0,
            extraction_metadata={
                'task_type': task_type_enum.value if isinstance(task_type_enum, TaskType) else task_type,
                'focus_ids': character_ids,
                'focus_type': 'multi_character',
                'extraction_time': datetime.utcnow().isoformat(),
            }
        )
        
        context.token_count = self._estimate_tokens(context)
        
        if context.token_count > max_tokens:
            context = self._prune_to_budget(context, max_tokens)
        
        return context
    
    def _extract_related_nodes(
        self,
        entity_id: str,
        requirements: ContextRequirement
    ) -> List[ScoredNode]:
        """Extract and score related nodes."""
        scored_nodes = []
        visited = {entity_id}
        
        neighbor_ids = self.graph.get_neighbors(
            entity_id,
            edge_types=requirements.edge_types,
            depth=requirements.max_depth
        )
        
        for nid in neighbor_ids:
            node = self.graph.get_node(nid)
            if not node:
                continue
            
            if node.type not in requirements.node_types:
                continue
            
            distance = self._get_distance(entity_id, nid)
            score = self._score_node(node, distance)
            
            if score >= requirements.importance_threshold:
                scored_nodes.append(ScoredNode(node=node, score=score, distance=distance))
                visited.add(nid)
        
        scored_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return scored_nodes
    
    def _score_node(self, node: Any, distance: int) -> float:
        """Calculate relevance score for a node."""
        importance = node.properties.get('importance', 0.5)
        recency = self._get_recency_score(node)
        
        distance_score = 1.0 / (distance + 1)
        
        score = (
            self.IMPORTANCE_WEIGHTS['node_importance'] * importance +
            self.IMPORTANCE_WEIGHTS['node_recency'] * recency +
            self.IMPORTANCE_WEIGHTS['distance'] * distance_score
        )
        
        return score
    
    def _get_recency_score(self, node: Any) -> float:
        """Calculate recency score (0-1) based on modification time."""
        try:
            modified = node.modified
            now = datetime.utcnow()
            age_days = (now - modified).days
            
            if age_days < 1:
                return 1.0
            elif age_days < 7:
                return 0.8
            elif age_days < 30:
                return 0.5
            else:
                return 0.2
        except:
            return 0.5
    
    def _get_distance(self, source_id: str, target_id: str) -> int:
        """Estimate graph distance between nodes."""
        path = self.graph.find_path(source_id, target_id, max_depth=3)
        return len(path) - 1 if path else 3
    
    def _select_top_nodes(
        self,
        scored_nodes: List[ScoredNode],
        max_nodes: int
    ) -> List[Any]:
        """Select top N nodes by score."""
        return [sn.node for sn in scored_nodes[:max_nodes]]
    
    def _extract_relationships(
        self,
        entity_id: str,
        related_ids: List[str],
        edge_types: List[str]
    ) -> List[Dict]:
        """Extract relationships between focus and related entities."""
        relationships = []
        
        edges = self.graph.get_edges(entity_id, edge_types=edge_types)
        
        all_ids = set(related_ids)
        all_ids.add(entity_id)
        
        for rid in related_ids[:10]:
            r_edges = self.graph.get_edges(rid, edge_types=edge_types)
            edges.extend(r_edges)
        
        seen = set()
        for edge in edges:
            key = (edge.source_id, edge.target_id, edge.edge_type)
            if key in seen:
                continue
            seen.add(key)
            
            if edge.source_id not in all_ids and edge.target_id not in all_ids:
                continue
            
            rel = {
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'edge_type': edge.edge_type,
                'weight': edge.weight,
                'context': edge.context,
                'bidirectional': edge.bidirectional,
            }
            
            source = self.graph.get_node(edge.source_id)
            target = self.graph.get_node(edge.target_id)
            
            if source:
                rel['source_name'] = source.properties.get('name', source.id)
                rel['source_type'] = source.type
            if target:
                rel['target_name'] = target.properties.get('name', target.id)
                rel['target_type'] = target.type
            
            relationships.append(rel)
        
        relationships.sort(
            key=lambda r: self.RELATIONSHIP_TYPE_PRIORITY.get(r['edge_type'], 0.5),
            reverse=True
        )
        
        return relationships[:20]
    
    def _extract_recent_events(
        self,
        entity_id: str,
        related_ids: List[str],
        limit: int = 10
    ) -> List[Any]:
        """Extract recent events involving the entity or related entities."""
        events = self.graph.list_nodes(node_type='event', limit=50)
        
        relevant_events = []
        all_ids = set(related_ids)
        all_ids.add(entity_id)
        
        for event in events:
            event_refs = self.graph.get_neighbors(event.id, depth=1)
            
            if entity_id in event_refs or any(rid in event_refs for rid in related_ids[:10]):
                relevant_events.append(event)
        
        relevant_events.sort(
            key=lambda e: e.modified if hasattr(e, 'modified') else datetime.min,
            reverse=True
        )
        
        return relevant_events[:limit]
    
    def _extract_conflicts(
        self,
        entity_id: str,
        related_ids: List[str]
    ) -> List[Dict]:
        """Extract active conflicts involving the entity."""
        conflicts = []
        
        conflict_edges = self.graph.get_edges(entity_id, edge_types=['conflict'])
        
        for edge in conflict_edges:
            other_id = edge.target_id if edge.source_id == entity_id else edge.source_id
            other = self.graph.get_node(other_id)
            
            conflict = {
                'parties': [
                    entity_id,
                    other_id
                ],
                'intensity': edge.weight or 0.5,
                'description': edge.context or f"Conflict with {other.properties.get('name', other_id) if other else other_id}",
                'edge_id': edge.id,
            }
            conflicts.append(conflict)
        
        for rid in related_ids[:5]:
            r_conflicts = self.graph.get_edges(rid, edge_types=['conflict'])
            for edge in r_conflicts:
                if edge.weight and edge.weight > 0.7:
                    conflicts.append({
                        'parties': [edge.source_id, edge.target_id],
                        'intensity': edge.weight,
                        'description': edge.context or 'High-intensity conflict',
                        'related_to_focus': True,
                    })
        
        return conflicts[:5]
    
    def _get_temporal_context(self, entity: Any) -> Dict[str, Any]:
        """Get temporal context for an entity."""
        return {
            'created': entity.created.isoformat() if hasattr(entity, 'created') else None,
            'modified': entity.modified.isoformat() if hasattr(entity, 'modified') else None,
            'generation': entity.properties.get('generation'),
            'timeline_position': entity.properties.get('timestamp'),
        }
    
    def _get_world_state_summary(self) -> Dict[str, Any]:
        """Get summary statistics of world state."""
        return {
            'total_characters': self.graph.count_nodes('character'),
            'total_factions': self.graph.count_nodes('faction'),
            'total_locations': self.graph.count_nodes('location'),
            'total_events': self.graph.count_nodes('event'),
            'total_species': self.graph.count_nodes('species'),
            'active_conflicts': len(self.graph.get_edges('', edge_types=['conflict'])) // 2,
        }
    
    def _estimate_tokens(self, context: ExtractedContext) -> int:
        """Estimate token count for context."""
        node_count = (
            1 +
            len(context.related_entities) +
            len(context.recent_events)
        )
        
        edge_count = len(context.relationships)
        
        return estimate_tokens(
            node_count,
            edge_count,
            include_temporal=bool(context.temporal_context)
        )
    
    def _prune_to_budget(
        self,
        context: ExtractedContext,
        max_tokens: int
    ) -> ExtractedContext:
        """Prune context to fit within token budget."""
        while context.token_count > max_tokens:
            if len(context.related_entities) > 5:
                context.related_entities.pop()
            elif len(context.recent_events) > 3:
                context.recent_events.pop()
            elif len(context.relationships) > 10:
                context.relationships.pop()
            elif len(context.active_conflicts) > 2:
                context.active_conflicts.pop()
            else:
                break
            
            context.token_count = self._estimate_tokens(context)
        
        return context
