"""
GraphCA Core - Cellular Automata on Graph Structures.

Operates CA rules directly on knowledge graph nodes and edges,
enabling emergent evolution of world state.
"""

from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
import math


class EvolutionPhase(str, Enum):
    """Phases of graph evolution."""
    INITIALIZE = "initialize"
    UPDATE_NODES = "update_nodes"
    UPDATE_EDGES = "update_edges"
    DETECT_EMERGENCE = "detect_emergence"
    CREATE_EVENTS = "create_events"
    CLEANUP = "cleanup"


@dataclass
class EvolutionState:
    """State of a single evolution step."""
    generation: int
    phase: EvolutionPhase
    nodes_updated: int
    edges_updated: int
    events_created: int
    patterns_detected: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'generation': self.generation,
            'phase': self.phase.value,
            'nodes_updated': self.nodes_updated,
            'edges_updated': self.edges_updated,
            'events_created': self.events_created,
            'patterns_detected': self.patterns_detected,
            'metadata': self.metadata,
        }


@dataclass
class EvolutionConfig:
    """Configuration for graph evolution."""
    max_generations: int = 100
    steps_per_generation: int = 1
    mutation_rate: float = 0.05
    extinction_threshold: int = 5
    population_cap: int = 10000
    event_threshold: float = 0.7
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)


class GraphCA:
    """
    Cellular Automata engine operating on graph structures.
    
    Unlike traditional CA where cells are arranged in a grid,
    GraphCA operates on nodes connected by edges. Each node's
    next state depends on its neighbors via edge relationships.
    """
    
    def __init__(
        self,
        graph: 'KnowledgeGraph',
        config: EvolutionConfig = None
    ):
        self.graph = graph
        self.config = config or EvolutionConfig()
        self.generation = 0
        self.history: List[EvolutionState] = []
        self._rules: Dict[str, Callable] = {}
        self._emergence_handlers: List[Callable] = []
    
    def register_rule(self, node_type: str, rule: Callable):
        """
        Register an evolution rule for a node type.
        
        Args:
            node_type: Type of node this rule applies to
            rule: Callable(node, neighbors, graph) -> updated_properties
        """
        self._rules[node_type] = rule
    
    def register_emergence_handler(self, handler: Callable):
        """
        Register a handler for emergence detection.
        
        Args:
            handler: Callable(graph, generation) -> List[event_data]
        """
        self._emergence_handlers.append(handler)
    
    def step(self) -> EvolutionState:
        """
        Execute one evolution step.
        
        Returns:
            EvolutionState describing what happened
        """
        state = EvolutionState(
            generation=self.generation,
            phase=EvolutionPhase.INITIALIZE,
            nodes_updated=0,
            edges_updated=0,
            events_created=0,
            patterns_detected=[],
        )
        
        state.phase = EvolutionPhase.UPDATE_NODES
        state.nodes_updated = self._update_all_nodes()
        
        state.phase = EvolutionPhase.UPDATE_EDGES
        state.edges_updated = self._update_all_edges()
        
        state.phase = EvolutionPhase.DETECT_EMERGENCE
        patterns = self._detect_emergence()
        state.patterns_detected = [p['type'] for p in patterns]
        
        state.phase = EvolutionPhase.CREATE_EVENTS
        state.events_created = self._create_events_from_patterns(patterns)
        
        state.phase = EvolutionPhase.CLEANUP
        self._cleanup()
        
        self.generation += 1
        self.history.append(state)
        
        return state
    
    def run(self, generations: int = None) -> List[EvolutionState]:
        """
        Run evolution for multiple generations.
        
        Args:
            generations: Number of generations (uses config if None)
        
        Returns:
            List of EvolutionState for each generation
        """
        generations = generations or self.config.max_generations
        results = []
        
        for _ in range(generations):
            state = self.step()
            results.append(state)
        
        return results
    
    def _update_all_nodes(self) -> int:
        """Update all nodes according to their rules."""
        updated = 0
        
        for node_type, rule in self._rules.items():
            nodes = self.graph.list_nodes(node_type=node_type)
            
            for node in nodes:
                neighbors = self._get_node_neighbors(node.id)
                
                try:
                    new_properties = rule(node, neighbors, self.graph)
                    
                    if new_properties:
                        self.graph.update_node(node.id, new_properties)
                        updated += 1
                except Exception as e:
                    pass
        
        return updated
    
    def _get_node_neighbors(self, node_id: str) -> Dict[str, List[Any]]:
        """Get neighbors organized by edge type."""
        neighbors = {
            'incoming': [],
            'outgoing': [],
            'bidirectional': [],
        }
        
        edges = self.graph.get_edges(node_id)
        
        for edge in edges:
            if edge.source_id == node_id:
                neighbor = self.graph.get_node(edge.target_id)
                if neighbor:
                    neighbors['outgoing'].append({
                        'node': neighbor,
                        'edge': edge,
                    })
            else:
                neighbor = self.graph.get_node(edge.source_id)
                if neighbor:
                    neighbors['incoming'].append({
                        'node': neighbor,
                        'edge': edge,
                    })
        
        return neighbors
    
    def _update_all_edges(self) -> int:
        """Update edge weights based on node interactions."""
        updated = 0
        
        edges_to_check = []
        
        species_nodes = self.graph.list_nodes(node_type='species')
        for species in species_nodes:
            edges = self.graph.get_edges(species.id, edge_types=['predation', 'competition', 'mutualism'])
            edges_to_check.extend(edges)
        
        for edge in edges_to_check:
            source = self.graph.get_node(edge.source_id)
            target = self.graph.get_node(edge.target_id)
            
            if not source or not target:
                continue
            
            new_weight = self._calculate_edge_weight(edge, source, target)
            
            if new_weight != edge.weight:
                self.graph.add_edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_type=edge.edge_type,
                    weight=new_weight,
                    context=edge.context,
                    bidirectional=edge.bidirectional,
                )
                updated += 1
        
        return updated
    
    def _calculate_edge_weight(
        self,
        edge: Any,
        source: Any,
        target: Any
    ) -> float:
        """Calculate new edge weight based on node properties."""
        base_weight = edge.weight or 0.5
        
        if edge.edge_type == 'predation':
            source_fitness = source.properties.get('fitness', 0.5)
            target_fitness = target.properties.get('fitness', 0.5)
            
            predation_efficiency = (source_fitness - target_fitness + 1) / 2
            new_weight = base_weight * 0.9 + predation_efficiency * 0.1
        
        elif edge.edge_type == 'competition':
            overlap = self._calculate_resource_overlap(source, target)
            new_weight = base_weight * 0.95 + overlap * 0.05
        
        elif edge.edge_type == 'mutualism':
            compatibility = self._calculate_compatibility(source, target)
            new_weight = base_weight * 0.95 + compatibility * 0.05
        
        else:
            mutation = random.gauss(0, self.config.mutation_rate)
            new_weight = base_weight + mutation
        
        return max(0.0, min(1.0, new_weight))
    
    def _calculate_resource_overlap(self, species_a: Any, species_b: Any) -> float:
        """Calculate resource overlap between two species."""
        biomes_a = set(species_a.properties.get('preferred_biomes', []))
        biomes_b = set(species_b.properties.get('preferred_biomes', []))
        
        if not biomes_a or not biomes_b:
            return 0.5
        
        intersection = len(biomes_a & biomes_b)
        union = len(biomes_a | biomes_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_compatibility(self, species_a: Any, species_b: Any) -> float:
        """Calculate mutualism compatibility between two species."""
        traits_a = species_a.properties.get('traits', {})
        traits_b = species_b.properties.get('traits', {})
        
        if not traits_a or not traits_b:
            return 0.5
        
        compatible = 0
        total = 0
        
        for trait, value_a in traits_a.items():
            if trait in traits_b:
                value_b = traits_b[trait]
                diff = abs(value_a - value_b)
                compatible += (1 - diff) if diff < 0.5 else 0
                total += 1
        
        return compatible / total if total > 0 else 0.5
    
    def _detect_emergence(self) -> List[Dict[str, Any]]:
        """Detect emergent patterns in the graph."""
        patterns = []
        
        for handler in self._emergence_handlers:
            try:
                detected = handler(self.graph, self.generation)
                patterns.extend(detected)
            except Exception as e:
                pass
        
        patterns.extend(self._detect_population_patterns())
        patterns.extend(self._detect_conflict_patterns())
        patterns.extend(self._detect_power_shifts())
        
        return patterns
    
    def _detect_population_patterns(self) -> List[Dict[str, Any]]:
        """Detect population-related emergence."""
        patterns = []
        
        species_nodes = self.graph.list_nodes(node_type='species')
        
        for species in species_nodes:
            population = species.properties.get('population', 0)
            
            if population <= self.config.extinction_threshold:
                patterns.append({
                    'type': 'extinction',
                    'entity_id': species.id,
                    'entity_name': species.properties.get('name', species.id),
                    'severity': 'high',
                    'details': {'final_population': population},
                })
            
            elif population > 500:
                patterns.append({
                    'type': 'population_boom',
                    'entity_id': species.id,
                    'entity_name': species.properties.get('name', species.id),
                    'severity': 'medium',
                    'details': {'population': population},
                })
        
        return patterns
    
    def _detect_conflict_patterns(self) -> List[Dict[str, Any]]:
        """Detect conflict escalation patterns."""
        patterns = []
        
        conflict_edges = []
        species_nodes = self.graph.list_nodes(node_type='species')
        for species in species_nodes:
            edges = self.graph.get_edges(species.id, edge_types=['conflict', 'competition'])
            conflict_edges.extend(edges)
        
        high_intensity = [e for e in conflict_edges if e.weight and e.weight > self.config.event_threshold]
        
        if len(high_intensity) >= 3:
            patterns.append({
                'type': 'conflict_escalation',
                'severity': 'high',
                'details': {
                    'conflict_count': len(high_intensity),
                    'average_intensity': sum(e.weight for e in high_intensity if e.weight) / len(high_intensity),
                },
            })
        
        return patterns
    
    def _detect_power_shifts(self) -> List[Dict[str, Any]]:
        """Detect power shift patterns in factions."""
        patterns = []
        
        faction_nodes = self.graph.list_nodes(node_type='faction')
        
        for faction in faction_nodes:
            power = faction.properties.get('power_level', 0.5)
            
            if power >= 0.9:
                patterns.append({
                    'type': 'dominant_power',
                    'entity_id': faction.id,
                    'entity_name': faction.properties.get('name', faction.id),
                    'severity': 'medium',
                    'details': {'power_level': power},
                })
            elif power <= 0.1:
                patterns.append({
                    'type': 'failing_faction',
                    'entity_id': faction.id,
                    'entity_name': faction.properties.get('name', faction.id),
                    'severity': 'medium',
                    'details': {'power_level': power},
                })
        
        return patterns
    
    def _create_events_from_patterns(self, patterns: List[Dict[str, Any]]) -> int:
        """Create event nodes from detected patterns."""
        events_created = 0
        
        for pattern in patterns:
            if pattern.get('severity') in ['high', 'medium']:
                event_data = self._pattern_to_event(pattern)
                
                if event_data:
                    try:
                        event_id = self.graph.add_node(
                            node_type='event',
                            properties=event_data['properties'],
                            tags=event_data['tags'],
                            source='ca_generated',
                        )
                        
                        if 'entity_id' in pattern:
                            self.graph.add_edge(
                                source_id=event_id,
                                target_id=pattern['entity_id'],
                                edge_type='references',
                                weight=1.0,
                                context=pattern['type'],
                            )
                        
                        events_created += 1
                    except Exception:
                        pass
        
        return events_created
    
    def _pattern_to_event(self, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a detected pattern to event data."""
        pattern_type = pattern.get('type', 'unknown')
        
        event_names = {
            'extinction': f"Extinction of {pattern.get('entity_name', 'Unknown')}",
            'population_boom': f"Population Surge: {pattern.get('entity_name', 'Unknown')}",
            'conflict_escalation': "Escalating Regional Conflicts",
            'dominant_power': f"Rise of {pattern.get('entity_name', 'Unknown')}",
            'failing_faction': f"Decline of {pattern.get('entity_name', 'Unknown')}",
        }
        
        return {
            'properties': {
                'name': event_names.get(pattern_type, f'Event: {pattern_type}'),
                'event_type': pattern_type,
                'generation': self.generation,
                'severity': pattern.get('severity', 'low'),
                'details': pattern.get('details', {}),
            },
            'tags': ['emergent', 'ca_generated', pattern_type],
        }
    
    def _cleanup(self):
        """Remove extinct entities and stale edges."""
        species_nodes = self.graph.list_nodes(node_type='species')
        
        for species in species_nodes:
            population = species.properties.get('population', 0)
            if population <= 0:
                self.graph.delete_node(species.id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current evolution statistics."""
        return {
            'generation': self.generation,
            'total_species': self.graph.count_nodes('species'),
            'total_characters': self.graph.count_nodes('character'),
            'total_factions': self.graph.count_nodes('faction'),
            'total_events': self.graph.count_nodes('event'),
            'history_length': len(self.history),
            'last_update': self.history[-1].to_dict() if self.history else None,
        }
