"""
CALab Bridge for GraphEngine.

Provides integration between CALab ecosystem/narrative CA systems
and the knowledge graph.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from GraphEngine.core.types import ModuleResult, RelationDef
from GraphEngine.modules.species_evolution import SpeciesEvolutionModule
from GraphEngine.modules.narrative_generation import NarrativeGenerationModule


class CALabBridge:
    """
    Bridge between CALab and GraphEngine.
    
    Converts CALab WorldState and world_dna data into knowledge graph
    nodes and edges.
    """
    
    def __init__(self, graph: 'KnowledgeGraph'):
        self.graph = graph
        
        self.species_module = SpeciesEvolutionModule()
        self.narrative_module = NarrativeGenerationModule()
        
        graph.register_module('species_evolution', self.species_module)
        graph.register_module('narrative_generation', self.narrative_module)
    
    def process_world_state(
        self,
        world_state: 'WorldState',
        generation: int = 0
    ) -> Dict[str, Any]:
        """
        Process a CALab WorldState into the knowledge graph.
        
        Args:
            world_state: WorldState from hybrid_ca_aggregator
            generation: Current generation number
        
        Returns:
            Dict with processing results
        """
        results = {
            'species_result': None,
            'narrative_result': None,
            'total_nodes_created': 0,
            'total_edges_created': 0
        }
        
        ecosystem_data = self._extract_ecosystem_data(world_state)
        if ecosystem_data:
            species_result = self.species_module.process(
                self.graph,
                ecosystem_state=ecosystem_data,
                generation=generation
            )
            results['species_result'] = species_result.to_dict()
            results['total_nodes_created'] += len(species_result.created_nodes)
            results['total_edges_created'] += len(species_result.created_edges)
        
        narrative_data = self._extract_narrative_data(world_state)
        if narrative_data:
            narrative_result = self.narrative_module.process(
                self.graph,
                narrative_state=narrative_data,
                generation=generation
            )
            results['narrative_result'] = narrative_result.to_dict()
            results['total_nodes_created'] += len(narrative_result.created_nodes)
            results['total_edges_created'] += len(narrative_result.created_edges)
        
        self._link_ecosystem_narrative(world_state, generation)
        
        return results
    
    def process_world_dna(
        self,
        world_dna: Dict[str, Any],
        generation: int = 0
    ) -> Dict[str, Any]:
        """
        Process world_dna from narrative_bridge into the knowledge graph.
        
        Args:
            world_dna: World DNA dict from narrative_bridge
            generation: Current generation number
        
        Returns:
            Dict with processing results
        """
        result = self.narrative_module.process(
            self.graph,
            world_dna=world_dna,
            generation=generation
        )
        
        return result.to_dict()
    
    def _extract_ecosystem_data(self, world_state: 'WorldState') -> Dict[str, Any]:
        """Extract ecosystem data from WorldState."""
        if not hasattr(world_state, 'ecosystem') or not world_state.ecosystem:
            return {}
        
        ecosystem = world_state.ecosystem
        
        species_list = []
        for species_id, species_data in ecosystem.get('species', {}).items():
            species_list.append({
                'id': species_id,
                'name': species_data.get('name', f'Species_{species_id}'),
                'type': species_data.get('type', 'unknown'),
                'population': species_data.get('population', 0),
                'fitness': species_data.get('fitness', 0.0),
                'traits': species_data.get('traits', {}),
                'center': species_data.get('center'),
                'territory_size': species_data.get('territory_size', 0)
            })
        
        relationships = []
        for rel in ecosystem.get('relationships', []):
            relationships.append({
                'source': rel.get('source'),
                'target': rel.get('target'),
                'type': rel.get('type'),
                'strength': rel.get('strength', 1.0)
            })
        
        return {
            'species': species_list,
            'relationships': relationships
        }
    
    def _extract_narrative_data(self, world_state: 'WorldState') -> Dict[str, Any]:
        """Extract narrative data from WorldState."""
        if not hasattr(world_state, 'narrative') or not world_state.narrative:
            return {}
        
        narrative = world_state.narrative
        
        return {
            'characters': narrative.get('characters', []),
            'factions': narrative.get('factions', []),
            'events': narrative.get('events', [])
        }
    
    def _link_ecosystem_narrative(self, world_state: 'WorldState', generation: int):
        """Create links between ecosystem and narrative elements."""
        if not hasattr(world_state, 'links'):
            return
        
        links = world_state.links if hasattr(world_state, 'links') else []
        
        for link in links:
            source = link.get('ecosystem_entity')
            target = link.get('narrative_entity')
            link_type = link.get('type', 'influences')
            
            if source and target:
                if self.graph.get_node(source) and self.graph.get_node(target):
                    self.graph.add_edge(
                        source_id=source,
                        target_id=target,
                        edge_type=link_type,
                        weight=link.get('weight', 0.5),
                        context=link.get('context', '')
                    )
    
    def get_ecosystem_summary(self) -> Dict[str, Any]:
        """Get summary of ecosystem-related nodes in the graph."""
        species_nodes = self.graph.list_nodes(node_type='species')
        event_nodes = self.graph.list_nodes(node_type='event')
        
        return {
            'species_count': len(species_nodes),
            'event_count': len(event_nodes),
            'total_population': sum(
                s.properties.get('population', 0) for s in species_nodes
            ),
            'avg_fitness': (
                sum(s.properties.get('fitness', 0) for s in species_nodes) / len(species_nodes)
                if species_nodes else 0
            )
        }
    
    def get_narrative_summary(self) -> Dict[str, Any]:
        """Get summary of narrative-related nodes in the graph."""
        return {
            'characters': self.graph.count_nodes('character'),
            'factions': self.graph.count_nodes('faction'),
            'events': self.graph.count_nodes('event'),
            'concepts': self.graph.count_nodes('concept')
        }
    
    def sync_to_obsidian(self) -> List[str]:
        """Export all CA-related nodes to Obsidian vault."""
        return self.graph.export_to_obsidian()
