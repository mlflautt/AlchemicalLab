"""
Species Evolution Processing Module.

Processes species nodes and ecological relationships based on
CA ecosystem evolution data.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import math

from GraphEngine.core.types import ModuleResult, RelationDef
from GraphEngine.modules.base_module import ProcessingModule


class SpeciesEvolutionModule(ProcessingModule):
    """
    Processing module for species evolution.
    
    Takes CA ecosystem data and:
    - Creates/updates species nodes
    - Generates ecological relationships (predation, competition, mutualism)
    - Tracks population and fitness changes
    - Detects evolutionary events
    """
    
    def __init__(self):
        super().__init__(
            name="species_evolution",
            description="Process CA ecosystem evolution into species nodes and relationships"
        )
        
        self.config = {
            'fitness_threshold': 0.1,
            'population_threshold': 10,
            'mutation_rate': 0.05,
            'track_lineage': True
        }
    
    def process(self, graph: 'KnowledgeGraph', **kwargs) -> ModuleResult:
        """
        Process ecosystem data into the knowledge graph.
        
        Args:
            graph: KnowledgeGraph instance
            ecosystem_state: Dict with species data from CA
            generation: Current generation number
        
        Returns:
            ModuleResult with created/modified nodes and edges
        """
        result = ModuleResult()
        
        ecosystem_state = kwargs.get('ecosystem_state', {})
        generation = kwargs.get('generation', 0)
        
        if not ecosystem_state:
            return result
        
        species_data = ecosystem_state.get('species', [])
        
        for species in species_data:
            node_id = self._process_species(graph, species, generation)
            if node_id:
                if generation == 0:
                    result.created_nodes.append(node_id)
                else:
                    result.modified_nodes.append(node_id)
        
        relationships = ecosystem_state.get('relationships', [])
        for rel in relationships:
            edge_key = self._process_relationship(graph, rel)
            if edge_key:
                result.created_edges.append(edge_key)
        
        events = self._detect_evolutionary_events(graph, ecosystem_state, generation)
        for event in events:
            node_id = self._create_event_node(graph, event)
            if node_id:
                result.created_nodes.append(node_id)
        
        result.metadata = {
            'generation': generation,
            'species_count': len(species_data),
            'relationship_count': len(relationships),
            'events_detected': len(events)
        }
        
        return result
    
    def _process_species(
        self,
        graph: 'KnowledgeGraph',
        species: Dict[str, Any],
        generation: int
    ) -> Optional[str]:
        """Create or update a species node."""
        species_id = species.get('id')
        if not species_id:
            return None
        
        name = species.get('name', f'Species_{species_id}')
        species_type = species.get('type', 'unknown')
        population = species.get('population', 0)
        fitness = species.get('fitness', 0.0)
        
        properties = {
            'name': name,
            'species_type': species_type,
            'population': population,
            'fitness': fitness,
            'generation': generation,
            'traits': species.get('traits', {}),
            'center': species.get('center'),
            'territory_size': species.get('territory_size', 0)
        }
        
        if 'preferred_biomes' in species:
            properties['preferred_biomes'] = species['preferred_biomes']
        if 'energy_efficiency' in species:
            properties['energy_efficiency'] = species['energy_efficiency']
        if 'reproduction_rate' in species:
            properties['reproduction_rate'] = species['reproduction_rate']
        
        existing = graph.get_node(species_id)
        
        if existing:
            graph.update_node(species_id, properties)
            return species_id
        else:
            return graph.add_node(
                node_type='species',
                properties=properties,
                tags=['ecosystem', 'ca-generated', f'gen-{generation}'],
                source='ca_generated',
                node_id=species_id
            )
    
    def _process_relationship(
        self,
        graph: 'KnowledgeGraph',
        relationship: Dict[str, Any]
    ) -> Optional[tuple]:
        """Create an ecological relationship edge."""
        source = relationship.get('source')
        target = relationship.get('target')
        rel_type = relationship.get('type')
        weight = relationship.get('strength', 1.0)
        
        if not all([source, target, rel_type]):
            return None
        
        edge_map = {
            'predation': 'predation',
            'competition': 'competition',
            'mutualism': 'mutualism',
            'symbiosis': 'mutualism'
        }
        
        edge_type = edge_map.get(rel_type, 'references')
        
        edge_id = graph.add_edge(
            source_id=source,
            target_id=target,
            edge_type=edge_type,
            weight=weight,
            context=relationship.get('context', ''),
            bidirectional=edge_type in ['competition', 'mutualism']
        )
        
        return (source, target, edge_type)
    
    def _detect_evolutionary_events(
        self,
        graph: 'KnowledgeGraph',
        ecosystem_state: Dict[str, Any],
        generation: int
    ) -> List[Dict[str, Any]]:
        """Detect significant evolutionary events."""
        events = []
        
        species_data = ecosystem_state.get('species', [])
        
        for species in species_data:
            species_id = species.get('id')
            population = species.get('population', 0)
            fitness = species.get('fitness', 0.0)
            
            existing = graph.get_node(species_id)
            if not existing:
                continue
            
            old_pop = existing.properties.get('population', 0)
            old_fitness = existing.properties.get('fitness', 0.0)
            
            if population <= self.config['population_threshold'] and old_pop > self.config['population_threshold']:
                events.append({
                    'type': 'extinction',
                    'species_id': species_id,
                    'species_name': species.get('name', species_id),
                    'generation': generation,
                    'final_population': population
                })
            
            if old_pop > 0:
                pop_change = (population - old_pop) / old_pop
                if pop_change > 0.5:
                    events.append({
                        'type': 'population_boom',
                        'species_id': species_id,
                        'species_name': species.get('name', species_id),
                        'generation': generation,
                        'change_percent': pop_change * 100
                    })
                elif pop_change < -0.3:
                    events.append({
                        'type': 'population_decline',
                        'species_id': species_id,
                        'species_name': species.get('name', species_id),
                        'generation': generation,
                        'change_percent': abs(pop_change) * 100
                    })
            
            if fitness > old_fitness + 0.1:
                events.append({
                    'type': 'fitness_increase',
                    'species_id': species_id,
                    'species_name': species.get('name', species_id),
                    'generation': generation,
                    'old_fitness': old_fitness,
                    'new_fitness': fitness
                })
        
        return events
    
    def _create_event_node(
        self,
        graph: 'KnowledgeGraph',
        event: Dict[str, Any]
    ) -> Optional[str]:
        """Create an event node for an evolutionary event."""
        event_type = event.get('type', 'unknown')
        species_id = event.get('species_id')
        generation = event.get('generation', 0)
        
        event_id = f"event_{event_type}_{species_id}_gen{generation}"
        
        name_map = {
            'extinction': f"Extinction of {event.get('species_name', 'Unknown')}",
            'population_boom': f"Population Boom: {event.get('species_name', 'Unknown')}",
            'population_decline': f"Population Decline: {event.get('species_name', 'Unknown')}",
            'fitness_increase': f"Fitness Increase: {event.get('species_name', 'Unknown')}"
        }
        
        properties = {
            'name': name_map.get(event_type, f'Event: {event_type}'),
            'event_type': event_type,
            'generation': generation,
            **{k: v for k, v in event.items() if k not in ['type', 'species_id', 'species_name']}
        }
        
        node_id = graph.add_node(
            node_type='event',
            properties=properties,
            tags=['evolutionary', 'ecosystem', event_type],
            source='module_created',
            node_id=event_id
        )
        
        if species_id:
            graph.add_edge(
                source_id=event_id,
                target_id=species_id,
                edge_type='references',
                weight=1.0,
                context=event_type
            )
        
        return node_id
