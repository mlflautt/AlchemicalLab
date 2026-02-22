"""
Emergence Detection for GraphCA.

Detects emergent patterns and behaviors in evolving graph structures.
"""

from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import math


@dataclass
class EmergencePattern:
    """A detected emergent pattern."""
    pattern_type: str
    entities: List[str]
    severity: str
    confidence: float
    details: Dict[str, Any]
    generation: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'pattern_type': self.pattern_type,
            'entities': self.entities,
            'severity': self.severity,
            'confidence': self.confidence,
            'details': self.details,
            'generation': self.generation,
            'timestamp': self.timestamp,
        }


class EmergenceDetector:
    """
    Detects emergent patterns in evolving knowledge graphs.
    
    Detects:
    - Population dynamics (extinction, boom, migration)
    - Conflict patterns (escalation, resolution, alliances)
    - Power shifts (faction rise/fall, dominance)
    - Network patterns (clusters, bridges, hubs)
    - Temporal patterns (cycles, trends)
    """
    
    def __init__(self, graph: 'KnowledgeGraph'):
        self.graph = graph
        self._history: List[Dict[str, Any]] = []
        self._thresholds = {
            'extinction': 5,
            'population_boom': 500,
            'conflict_escalation': 0.7,
            'power_shift': 0.2,
            'cluster_density': 0.6,
        }
    
    def detect_all(self, generation: int = 0) -> List[EmergencePattern]:
        """Run all detection methods."""
        patterns = []
        
        patterns.extend(self.detect_population_dynamics(generation))
        patterns.extend(self.detect_conflict_patterns(generation))
        patterns.extend(self.detect_power_shifts(generation))
        patterns.extend(self.detect_network_patterns(generation))
        patterns.extend(self.detect_temporal_patterns(generation))
        
        self._history.append({
            'generation': generation,
            'pattern_count': len(patterns),
            'pattern_types': [p.pattern_type for p in patterns],
        })
        
        return patterns
    
    def detect_population_dynamics(self, generation: int = 0) -> List[EmergencePattern]:
        """Detect population-related emergence."""
        patterns = []
        species_nodes = self.graph.list_nodes(node_type='species')
        
        total_population = 0
        population_distribution = defaultdict(int)
        
        for species in species_nodes:
            pop = species.properties.get('population', 0)
            total_population += pop
            population_distribution[species.id] = pop
            
            if pop <= self._thresholds['extinction']:
                patterns.append(EmergencePattern(
                    pattern_type='extinction_imminent',
                    entities=[species.id],
                    severity='high',
                    confidence=0.9,
                    details={
                        'species_name': species.properties.get('name', species.id),
                        'population': pop,
                        'fitness': species.properties.get('fitness', 0),
                    },
                    generation=generation,
                ))
            
            elif pop >= self._thresholds['population_boom']:
                patterns.append(EmergencePattern(
                    pattern_type='population_explosion',
                    entities=[species.id],
                    severity='medium',
                    confidence=0.8,
                    details={
                        'species_name': species.properties.get('name', species.id),
                        'population': pop,
                        'fitness': species.properties.get('fitness', 0),
                    },
                    generation=generation,
                ))
        
        if len(population_distribution) > 0:
            pops = list(population_distribution.values())
            mean_pop = sum(pops) / len(pops)
            variance = sum((p - mean_pop) ** 2 for p in pops) / len(pops)
            std_dev = math.sqrt(variance)
            
            if std_dev > mean_pop:
                patterns.append(EmergencePattern(
                    pattern_type='population_inequality',
                    entities=list(population_distribution.keys()),
                    severity='low',
                    confidence=0.7,
                    details={
                        'mean_population': mean_pop,
                        'std_deviation': std_dev,
                        'total_species': len(species_nodes),
                    },
                    generation=generation,
                ))
        
        return patterns
    
    def detect_conflict_patterns(self, generation: int = 0) -> List[EmergencePattern]:
        """Detect conflict-related emergence."""
        patterns = []
        
        conflict_edges = []
        all_edges = []
        
        species_nodes = self.graph.list_nodes(node_type='species')
        faction_nodes = self.graph.list_nodes(node_type='faction')
        all_nodes = species_nodes + faction_nodes
        
        for node in all_nodes:
            edges = self.graph.get_edges(node.id, edge_types=['conflict', 'competition'])
            conflict_edges.extend(edges)
        
        unique_conflicts = set()
        for edge in conflict_edges:
            key = tuple(sorted([edge.source_id, edge.target_id]))
            unique_conflicts.add(key)
        
        high_intensity_conflicts = [
            e for e in conflict_edges
            if e.weight and e.weight >= self._thresholds['conflict_escalation']
        ]
        
        if len(high_intensity_conflicts) >= 3:
            involved_entities = set()
            for edge in high_intensity_conflicts:
                involved_entities.add(edge.source_id)
                involved_entities.add(edge.target_id)
            
            patterns.append(EmergencePattern(
                pattern_type='conflict_escalation',
                entities=list(involved_entities),
                severity='high',
                confidence=0.85,
                details={
                    'high_intensity_count': len(high_intensity_conflicts),
                    'total_conflicts': len(unique_conflicts),
                    'average_intensity': sum(e.weight or 0 for e in high_intensity_conflicts) / len(high_intensity_conflicts),
                },
                generation=generation,
            ))
        
        alliances = []
        for node in all_nodes:
            alliance_edges = self.graph.get_edges(node.id, edge_types=['alliance'])
            alliances.extend(alliance_edges)
        
        if len(alliances) > len(conflict_edges):
            patterns.append(EmergencePattern(
                pattern_type='peaceful_period',
                entities=[],
                severity='low',
                confidence=0.7,
                details={
                    'alliance_count': len(alliances),
                    'conflict_count': len(conflict_edges),
                },
                generation=generation,
            ))
        
        return patterns
    
    def detect_power_shifts(self, generation: int = 0) -> List[EmergencePattern]:
        """Detect power-related emergence in factions."""
        patterns = []
        faction_nodes = self.graph.list_nodes(node_type='faction')
        
        dominant_factions = []
        declining_factions = []
        balanced_factions = []
        
        for faction in faction_nodes:
            power = faction.properties.get('power_level', 0.5)
            
            if power >= 0.85:
                dominant_factions.append(faction)
            elif power <= 0.15:
                declining_factions.append(faction)
            else:
                balanced_factions.append(faction)
        
        if len(dominant_factions) == 1:
            faction = dominant_factions[0]
            patterns.append(EmergencePattern(
                pattern_type='hegemony',
                entities=[faction.id],
                severity='medium',
                confidence=0.8,
                details={
                    'faction_name': faction.properties.get('name', faction.id),
                    'power_level': faction.properties.get('power_level', 0),
                },
                generation=generation,
            ))
        
        elif len(dominant_factions) >= 2:
            patterns.append(EmergencePattern(
                pattern_type='bipolar_competition',
                entities=[f.id for f in dominant_factions],
                severity='medium',
                confidence=0.75,
                details={
                    'dominant_count': len(dominant_factions),
                    'powers': [f.properties.get('power_level', 0) for f in dominant_factions],
                },
                generation=generation,
            ))
        
        if len(declining_factions) > len(faction_nodes) * 0.3:
            patterns.append(EmergencePattern(
                pattern_type='systemic_collapse',
                entities=[f.id for f in declining_factions],
                severity='high',
                confidence=0.7,
                details={
                    'declining_count': len(declining_factions),
                    'total_factions': len(faction_nodes),
                },
                generation=generation,
            ))
        
        return patterns
    
    def detect_network_patterns(self, generation: int = 0) -> List[EmergencePattern]:
        """Detect structural patterns in the graph."""
        patterns = []
        
        all_nodes = (
            self.graph.list_nodes(node_type='character') +
            self.graph.list_nodes(node_type='faction')
        )
        
        degree_centrality = {}
        for node in all_nodes:
            edges = self.graph.get_edges(node.id)
            degree_centrality[node.id] = len(edges)
        
        if degree_centrality:
            max_degree = max(degree_centrality.values())
            avg_degree = sum(degree_centrality.values()) / len(degree_centrality)
            
            hubs = [
                node_id for node_id, degree in degree_centrality.items()
                if degree > avg_degree * 2
            ]
            
            if hubs:
                patterns.append(EmergencePattern(
                    pattern_type='hub_entities',
                    entities=hubs,
                    severity='low',
                    confidence=0.8,
                    details={
                        'hub_count': len(hubs),
                        'average_degree': avg_degree,
                        'max_degree': max_degree,
                    },
                    generation=generation,
                ))
        
        clusters = self._detect_clusters(all_nodes)
        if len(clusters) > 1:
            patterns.append(EmergencePattern(
                pattern_type='clustered_network',
                entities=[node_id for cluster in clusters for node_id in cluster[:3]],
                severity='low',
                confidence=0.7,
                details={
                    'cluster_count': len(clusters),
                    'largest_cluster': max(len(c) for c in clusters),
                    'smallest_cluster': min(len(c) for c in clusters),
                },
                generation=generation,
            ))
        
        return patterns
    
    def detect_temporal_patterns(self, generation: int = 0) -> List[EmergencePattern]:
        """Detect patterns over time."""
        patterns = []
        
        if len(self._history) < 3:
            return patterns
        
        recent = self._history[-3:]
        
        extinction_history = [
            h for h in recent
            if 'extinction_imminent' in h.get('pattern_types', [])
        ]
        
        if len(extinction_history) >= 2:
            patterns.append(EmergencePattern(
                pattern_type='mass_extinction_trend',
                entities=[],
                severity='high',
                confidence=0.75,
                details={
                    'consecutive_generations': len(extinction_history),
                },
                generation=generation,
            ))
        
        conflict_history = [
            h for h in recent
            if 'conflict_escalation' in h.get('pattern_types', [])
        ]
        
        if len(conflict_history) >= 3:
            patterns.append(EmergencePattern(
                pattern_type='prolonged_war',
                entities=[],
                severity='high',
                confidence=0.8,
                details={
                    'consecutive_generations': len(conflict_history),
                },
                generation=generation,
            ))
        
        return patterns
    
    def _detect_clusters(self, nodes: List[Any]) -> List[List[str]]:
        """Detect clusters using simple connected components."""
        visited = set()
        clusters = []
        
        for node in nodes:
            if node.id in visited:
                continue
            
            cluster = self._bfs_cluster(node.id, visited)
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _bfs_cluster(self, start_id: str, visited: Set[str]) -> List[str]:
        """BFS to find connected component."""
        from collections import deque
        
        cluster = []
        queue = deque([start_id])
        
        while queue:
            node_id = queue.popleft()
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            cluster.append(node_id)
            
            neighbors = self.graph.get_neighbors(node_id, depth=1)
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    queue.append(neighbor_id)
        
        return cluster
    
    def get_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get detection history."""
        return self._history[-last_n:]
    
    def set_threshold(self, name: str, value: float):
        """Set a detection threshold."""
        self._thresholds[name] = value


def create_emergence_handler(detector: EmergenceDetector) -> Callable:
    """
    Create an emergence handler for GraphCA.
    
    Args:
        detector: EmergenceDetector instance
    
    Returns:
        Handler function for GraphCA.register_emergence_handler
    """
    def handler(graph: 'KnowledgeGraph', generation: int) -> List[Dict[str, Any]]:
        detector.graph = graph
        patterns = detector.detect_all(generation)
        return [p.to_dict() for p in patterns]
    
    return handler
