"""
Hybrid CA Aggregator - Combines Ecosystem and Narrative Emergence Systems
=========================================================================

Aggregates outputs from multiple CALab systems into a unified WorldState
that can be used for cross-lab integration with StoryLab.

Systems Combined:
- EcosystemEvolution: Species, food webs, ecological dynamics
- NarrativeEmergence: Characters, locations, factions, story arcs
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json


@dataclass
class SpeciesData:
    species_id: int
    name: str
    species_type: str
    center: Tuple[int, int]
    population: int
    traits: Dict[str, float]
    preferred_biomes: List[str]
    fitness: float
    generation_alive: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CharacterData:
    element_id: int
    name: str
    element_type: str
    center: Tuple[int, int]
    properties: Dict[str, Any]
    backstory: List[str]
    relationships: Set[int]
    narrative_weight: float
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['relationships'] = list(self.relationships)
        return d


@dataclass
class LocationData:
    element_id: int
    name: str
    element_type: str
    center: Tuple[int, int]
    properties: Dict[str, Any]
    atmosphere: str
    resources: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FactionData:
    element_id: int
    name: str
    ideology: str
    methods: str
    goal: str
    members: List[int]
    territories: List[int]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StoryArcData:
    arc_id: int
    arc_type: str
    participants: List[int]
    story_beats: List[str]
    tension_level: float
    completion: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EcologicalRelationshipData:
    predator_id: int
    prey_id: int
    relationship_type: str
    strength: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class NarrativeRelationshipData:
    source_id: int
    target_id: int
    relationship_type: str
    strength: float
    story_events: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WorldState:
    generation: int
    world_size: Tuple[int, int]
    
    species: Dict[int, SpeciesData] = field(default_factory=dict)
    characters: Dict[int, CharacterData] = field(default_factory=dict)
    locations: Dict[int, LocationData] = field(default_factory=dict)
    factions: Dict[int, FactionData] = field(default_factory=dict)
    story_arcs: Dict[int, StoryArcData] = field(default_factory=dict)
    
    ecological_relationships: List[EcologicalRelationshipData] = field(default_factory=list)
    narrative_relationships: List[NarrativeRelationshipData] = field(default_factory=list)
    
    biomes: Dict[str, Dict] = field(default_factory=dict)
    regions: Dict[str, Dict] = field(default_factory=dict)
    
    ca_grid: Optional[np.ndarray] = None
    
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = {
            'generation': self.generation,
            'world_size': self.world_size,
            'species': {k: v.to_dict() for k, v in self.species.items()},
            'characters': {k: v.to_dict() for k, v in self.characters.items()},
            'locations': {k: v.to_dict() for k, v in self.locations.items()},
            'factions': {k: v.to_dict() for k, v in self.factions.items()},
            'story_arcs': {k: v.to_dict() for k, v in self.story_arcs.items()},
            'ecological_relationships': [r.to_dict() for r in self.ecological_relationships],
            'narrative_relationships': [r.to_dict() for r in self.narrative_relationships],
            'biomes': self.biomes,
            'regions': self.regions,
            'stats': self.stats
        }
        if self.ca_grid is not None:
            result['ca_grid_shape'] = self.ca_grid.shape
            result['ca_density'] = float(np.mean(self.ca_grid))
        return result
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class HybridCAAggregator:
    """
    Aggregates ecosystem and narrative emergence systems into unified WorldState.
    """
    
    def __init__(self, world_size: Tuple[int, int] = (150, 150)):
        self.world_size = world_size
        self.ecosystem = None
        self.narrative_system = None
        self.world_state = WorldState(
            generation=0,
            world_size=world_size
        )
        
    def initialize(self, density: float = 0.3, seed: int = 42):
        from CALab.ecosystem_evolution import EcosystemSimulation
        from CALab.narrative_emergence import WorldBuildingSystem
        
        self.ecosystem = EcosystemSimulation(world_size=self.world_size)
        self.ecosystem.initialize_ecosystem(density=density, seed=seed)
        
        self.narrative_system = WorldBuildingSystem(world_size=self.world_size)
        self.narrative_system.initialize_world(density=density, seed=seed + 1000)
        
        self._sync_initial_state()
        
    def _sync_initial_state(self):
        self.world_state.biomes = {
            zone.biome_type: {
                'center': zone.center,
                'radius': zone.radius,
                'temperature': zone.temperature,
                'moisture': zone.moisture,
                'productivity': zone.nutrients
            }
            for zone in self.ecosystem.environmental_zones.values()
        }
        
        self.world_state.regions = {
            name: {
                'center': data['center'],
                'radius': data['radius'],
                'element_bias': data['element_bias']
            }
            for name, data in self.narrative_system.regions.items()
        }
        
    def step(self):
        self.ecosystem.step_ecosystem()
        
        self.narrative_system.step_ca()
        self.narrative_system.detect_and_generate_narrative_elements()
        self.narrative_system.evolve_relationships()
        
        self._aggregate_state()
        self.world_state.generation += 1
        
    def run(self, generations: int, progress_callback=None):
        for i in range(generations):
            self.step()
            if progress_callback and i % 25 == 0:
                progress_callback(i, self.world_state)
        return self.world_state
    
    def _aggregate_state(self):
        ws = self.world_state
        
        ws.species.clear()
        for species in self.ecosystem.species.values():
            ws.species[species.species_id] = SpeciesData(
                species_id=species.species_id,
                name=species.species_name,
                species_type=species.species_type,
                center=species.center,
                population=species.population,
                traits=dict(species.traits),
                preferred_biomes=species.preferred_biomes,
                fitness=species.fitness,
                generation_alive=species.generation_alive
            )
        
        ws.characters.clear()
        ws.locations.clear()
        ws.factions.clear()
        
        for element in self.narrative_system.narrative_elements.values():
            if element.element_type == 'character':
                ws.characters[element.element_id] = CharacterData(
                    element_id=element.element_id,
                    name=element.name,
                    element_type=element.element_type,
                    center=element.center,
                    properties=element.properties,
                    backstory=element.backstory,
                    relationships=set(element.relationships),
                    narrative_weight=element.narrative_weight
                )
            elif element.element_type == 'location':
                ws.locations[element.element_id] = LocationData(
                    element_id=element.element_id,
                    name=element.name,
                    element_type=element.element_type,
                    center=element.center,
                    properties=element.properties,
                    atmosphere=element.properties.get('atmosphere', 'unknown'),
                    resources=element.properties.get('resources', 'unknown')
                )
            elif element.element_type == 'faction':
                ws.factions[element.element_id] = FactionData(
                    element_id=element.element_id,
                    name=element.name,
                    ideology=element.properties.get('ideology', 'unknown'),
                    methods=element.properties.get('methods', 'unknown'),
                    goal=element.properties.get('goal', 'unknown'),
                    members=[],
                    territories=[]
                )
        
        ws.story_arcs.clear()
        for arc in self.narrative_system.story_arcs.values():
            ws.story_arcs[arc.arc_id] = StoryArcData(
                arc_id=arc.arc_id,
                arc_type=arc.arc_type,
                participants=list(arc.participants),
                story_beats=arc.story_beats,
                tension_level=arc.tension_level,
                completion=arc.completion
            )
        
        ws.ecological_relationships.clear()
        for rel in self.ecosystem.relationships.values():
            ws.ecological_relationships.append(EcologicalRelationshipData(
                predator_id=rel.predator_id,
                prey_id=rel.prey_id,
                relationship_type=rel.relationship_type,
                strength=rel.strength
            ))
        
        ws.narrative_relationships.clear()
        for rel in self.narrative_system.relationships.values():
            ws.narrative_relationships.append(NarrativeRelationshipData(
                source_id=rel.source_id,
                target_id=rel.target_id,
                relationship_type=rel.relationship_type,
                strength=rel.strength,
                story_events=rel.story_events
            ))
        
        ws.ca_grid = self.ecosystem.ca_grid.copy()
        
        ws.stats = {
            'ecosystem': self.ecosystem.get_ecosystem_stats(),
            'narrative': self.narrative_system.get_world_stats()
        }
        
    def get_world_state(self) -> WorldState:
        return self.world_state
    
    def get_combined_network(self):
        import networkx as nx
        
        G = nx.Graph()
        
        for species_id, species in self.world_state.species.items():
            G.add_node(
                f"species_{species_id}",
                node_type='species',
                species_type=species.species_type,
                name=species.name,
                population=species.population
            )
        
        for char_id, char in self.world_state.characters.items():
            G.add_node(
                f"character_{char_id}",
                node_type='character',
                name=char.name,
                narrative_weight=char.narrative_weight
            )
        
        for loc_id, loc in self.world_state.locations.items():
            G.add_node(
                f"location_{loc_id}",
                node_type='location',
                name=loc.name,
                atmosphere=loc.atmosphere
            )
        
        for faction_id, faction in self.world_state.factions.items():
            G.add_node(
                f"faction_{faction_id}",
                node_type='faction',
                name=faction.name,
                ideology=faction.ideology
            )
        
        for rel in self.world_state.ecological_relationships:
            G.add_edge(
                f"species_{rel.predator_id}",
                f"species_{rel.prey_id}",
                edge_type='ecological',
                relationship_type=rel.relationship_type,
                strength=rel.strength
            )
        
        for rel in self.world_state.narrative_relationships:
            source_type = None
            target_type = None
            
            if rel.source_id in self.world_state.characters:
                source_type = 'character'
            elif rel.source_id in self.world_state.locations:
                source_type = 'location'
            elif rel.source_id in self.world_state.factions:
                source_type = 'faction'
                
            if rel.target_id in self.world_state.characters:
                target_type = 'character'
            elif rel.target_id in self.world_state.locations:
                target_type = 'location'
            elif rel.target_id in self.world_state.factions:
                target_type = 'faction'
            
            if source_type and target_type:
                G.add_edge(
                    f"{source_type}_{rel.source_id}",
                    f"{target_type}_{rel.target_id}",
                    edge_type='narrative',
                    relationship_type=rel.relationship_type,
                    strength=rel.strength
                )
        
        return G
    
    def cross_map_entities(self) -> Dict[str, List[Dict]]:
        """
        Create mappings between ecological and narrative entities.
        This enables species to become character inspiration, etc.
        """
        mappings = {
            'species_to_character': [],
            'location_to_biome': [],
            'faction_to_species': []
        }
        
        for species_id, species in self.world_state.species.items():
            for char_id, char in self.world_state.characters.items():
                dist = np.sqrt(
                    (species.center[0] - char.center[0])**2 +
                    (species.center[1] - char.center[1])**2
                )
                if dist < 20:
                    mappings['species_to_character'].append({
                        'species_id': species_id,
                        'species_name': species.name,
                        'species_type': species.species_type,
                        'character_id': char_id,
                        'character_name': char.name,
                        'distance': dist
                    })
        
        for loc_id, loc in self.world_state.locations.items():
            for biome_name, biome_data in self.world_state.biomes.items():
                dist = np.sqrt(
                    (loc.center[0] - biome_data['center'][0])**2 +
                    (loc.center[1] - biome_data['center'][1])**2
                )
                if dist < biome_data['radius']:
                    mappings['location_to_biome'].append({
                        'location_id': loc_id,
                        'location_name': loc.name,
                        'biome_name': biome_name,
                        'distance': dist
                    })
        
        return mappings


def run_hybrid_simulation(
    world_size: Tuple[int, int] = (100, 100),
    generations: int = 100,
    density: float = 0.3,
    seed: int = 42,
    progress_callback=None
) -> WorldState:
    """
    Convenience function to run a complete hybrid simulation.
    """
    aggregator = HybridCAAggregator(world_size=world_size)
    aggregator.initialize(density=density, seed=seed)
    
    def default_progress(gen, state):
        print(f"Gen {gen}: {len(state.species)} species, "
              f"{len(state.characters)} characters, "
              f"{len(state.locations)} locations, "
              f"{len(state.story_arcs)} story arcs")
    
    callback = progress_callback or default_progress
    return aggregator.run(generations, progress_callback=callback)


if __name__ == "__main__":
    print("Hybrid CA Aggregator - Cross-Lab Integration Test")
    print("=" * 50)
    
    world_state = run_hybrid_simulation(
        world_size=(80, 80),
        generations=50,
        density=0.3,
        seed=42
    )
    
    print("\n--- Final World State ---")
    print(f"Generation: {world_state.generation}")
    print(f"Species: {len(world_state.species)}")
    print(f"Characters: {len(world_state.characters)}")
    print(f"Locations: {len(world_state.locations)}")
    print(f"Factions: {len(world_state.factions)}")
    print(f"Story Arcs: {len(world_state.story_arcs)}")
    print(f"Ecological Relationships: {len(world_state.ecological_relationships)}")
    print(f"Narrative Relationships: {len(world_state.narrative_relationships)}")
    
    print("\n--- Sample Species ---")
    for species in list(world_state.species.values())[:3]:
        print(f"  {species.name} ({species.species_type}): pop={species.population}")
    
    print("\n--- Sample Characters ---")
    for char in list(world_state.characters.values())[:3]:
        print(f"  {char.name}: {char.properties.get('class', 'unknown')}")
    
    print("\n--- Cross-Entity Mappings ---")
    aggregator = HybridCAAggregator()
    aggregator.world_state = world_state
    mappings = aggregator.cross_map_entities()
    print(f"Species-Character pairs: {len(mappings['species_to_character'])}")
    print(f"Location-Biome pairs: {len(mappings['location_to_biome'])}")
