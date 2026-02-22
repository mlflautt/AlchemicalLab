"""
CA Adapter for StoryLab - Direct Integration Layer
===================================================

Provides seamless integration between CALab's hybrid systems and
StoryLab's LLM-based narrative generation.

Enables:
- CA WorldState → StoryLab world_dna conversion
- Direct generation from CA simulations
- Entity extraction and embedding sync
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class CAStoryContext:
    context_id: str
    world_state: Any
    world_dna: str
    entities: Dict[str, List[Dict]] = field(default_factory=dict)
    relationships: List[Dict] = field(default_factory=list)
    generation_metadata: Dict = field(default_factory=dict)


class CAAdapter:
    """
    Adapter for integrating CA systems with StoryLab.
    """
    
    def __init__(
        self,
        model_path: str = "../qwen_q2.gguf",
        ngl: int = 35,
        auto_initialize_db: bool = True
    ):
        self.model_path = model_path
        self.ngl = ngl
        self.db = None
        self._story_generator = None
        self._world_dna_generator = None
        
        if auto_initialize_db:
            self._init_db()
    
    def _init_db(self):
        try:
            from StoryLab.db_manager import StoryDBManager
            self.db = StoryDBManager()
        except Exception as e:
            print(f"Warning: Could not initialize StoryDB: {e}")
            self.db = None
    
    @property
    def story_generator(self):
        if self._story_generator is None:
            from StoryLab.story_generator import generate_story_idea
            self._story_generator = generate_story_idea
        return self._story_generator
    
    @property
    def world_dna_generator(self):
        if self._world_dna_generator is None:
            from CALab.narrative_bridge import WorldDNAGenerator
            self._world_dna_generator = WorldDNAGenerator()
        return self._world_dna_generator
    
    def load_world_state(
        self,
        source: Any,
        source_type: str = "world_state"
    ) -> CAStoryContext:
        """
        Load a WorldState from various sources.
        
        Args:
            source: WorldState object, file path, or dict
            source_type: 'world_state', 'file', or 'dict'
        
        Returns:
            CAStoryContext with world state and DNA
        """
        if source_type == "world_state":
            world_state = source
        elif source_type == "file":
            import json
            with open(source, 'r') as f:
                data = json.load(f)
            world_state = self._dict_to_world_state(data)
        elif source_type == "dict":
            world_state = self._dict_to_world_state(source)
        else:
            raise ValueError(f"Unknown source_type: {source_type}")
        
        world_dna = self.world_dna_generator.generate_world_dna(world_state)
        
        entities = self._extract_entities(world_state)
        relationships = self._extract_relationships(world_state)
        
        context_id = f"ca_context_{int(time.time())}"
        
        return CAStoryContext(
            context_id=context_id,
            world_state=world_state,
            world_dna=world_dna,
            entities=entities,
            relationships=relationships,
            generation_metadata={
                'generation': world_state.generation,
                'world_size': world_state.world_size,
                'num_species': len(world_state.species),
                'num_characters': len(world_state.characters),
                'num_locations': len(world_state.locations)
            }
        )
    
    def _dict_to_world_state(self, data: Dict):
        from CALab.hybrid_ca_aggregator import WorldState, SpeciesData, CharacterData, LocationData
        
        ws = WorldState(
            generation=data.get('generation', 0),
            world_size=tuple(data.get('world_size', (100, 100)))
        )
        
        for sid, sdata in data.get('species', {}).items():
            ws.species[int(sid)] = SpeciesData(**sdata)
        
        for cid, cdata in data.get('characters', {}).items():
            ws.characters[int(cid)] = CharacterData(**cdata)
        
        for lid, ldata in data.get('locations', {}).items():
            ws.locations[int(lid)] = LocationData(**ldata)
        
        ws.stats = data.get('stats', {})
        
        return ws
    
    def _extract_entities(self, world_state) -> Dict[str, List[Dict]]:
        """Extract entities from world state for embedding."""
        entities = {
            'species': [],
            'characters': [],
            'locations': [],
            'factions': [],
            'story_arcs': []
        }
        
        for species in world_state.species.values():
            entities['species'].append({
                'id': species.species_id,
                'name': species.name,
                'type': species.species_type,
                'traits': species.traits,
                'population': species.population,
                'description': f"{species.name} is a {species.species_type} species with {species.population} individuals"
            })
        
        for char in world_state.characters.values():
            entities['characters'].append({
                'id': char.element_id,
                'name': char.name,
                'properties': char.properties,
                'backstory': char.backstory,
                'description': ' '.join(char.backstory) if char.backstory else char.name
            })
        
        for loc in world_state.locations.values():
            entities['locations'].append({
                'id': loc.element_id,
                'name': loc.name,
                'atmosphere': loc.atmosphere,
                'resources': loc.resources,
                'description': f"{loc.name}: a {loc.atmosphere} location known for {loc.resources}"
            })
        
        return entities
    
    def _extract_relationships(self, world_state) -> List[Dict]:
        """Extract relationships from world state."""
        relationships = []
        
        for rel in world_state.ecological_relationships:
            pred = world_state.species.get(rel.predator_id)
            prey = world_state.species.get(rel.prey_id)
            if pred and prey:
                relationships.append({
                    'type': 'ecological',
                    'relationship_type': rel.relationship_type,
                    'source': pred.name,
                    'target': prey.name,
                    'strength': rel.strength,
                    'description': f"{pred.name} ({rel.relationship_type}) {prey.name}"
                })
        
        for rel in world_state.narrative_relationships:
            relationships.append({
                'type': 'narrative',
                'relationship_type': rel.relationship_type,
                'source_id': rel.source_id,
                'target_id': rel.target_id,
                'strength': rel.strength,
                'story_events': rel.story_events
            })
        
        return relationships
    
    def generate_story(
        self,
        context: CAStoryContext,
        multi_step: bool = False,
        theme: str = "",
        use_news: bool = False,
        use_books: bool = False,
        use_images: bool = False,
        use_tts: bool = False,
        max_retries: int = 3
    ) -> Tuple[str, Dict, Dict]:
        """
        Generate a story from CA context using StoryLab.
        
        Returns:
            Tuple of (story_text, critique, extra_info)
        """
        return self.story_generator(
            context.world_dna,
            model_path=self.model_path,
            ngl=self.ngl,
            max_retries=max_retries,
            db=self.db,
            multi_step=multi_step,
            theme=theme,
            use_news=use_news,
            use_books=use_books,
            use_images=use_images,
            use_tts=use_tts
        )
    
    def generate_story_from_world_state(
        self,
        world_state,
        **kwargs
    ) -> Tuple[str, Dict, Dict]:
        """
        Convenience method: WorldState → Story in one call.
        """
        context = self.load_world_state(world_state)
        return self.generate_story(context, **kwargs)
    
    def sync_embeddings_to_db(self, context: CAStoryContext):
        """
        Sync CA-derived entities to ChromaDB for vector search.
        """
        if self.db is None:
            print("Warning: No database connection for embedding sync")
            return
        
        from StoryLab.db_manager import StoryDBManager
        
        for entity_type, entities in context.entities.items():
            for entity in entities:
                if entity.get('description'):
                    try:
                        entity_id = f"{entity_type}_{entity['id']}"
                        self.db.add_story(
                            entity_id,
                            entity['description'],
                            {
                                'type': entity_type,
                                'source': 'ca_generated',
                                **{k: v for k, v in entity.items() if k != 'description'}
                            }
                        )
                    except Exception as e:
                        print(f"Warning: Could not embed {entity_type}/{entity.get('id')}: {e}")
    
    def run_full_pipeline(
        self,
        world_size: Tuple[int, int] = (100, 100),
        generations: int = 50,
        density: float = 0.3,
        seed: int = 42,
        world_name: str = "Emergent World",
        genre: str = "Fantasy",
        **story_kwargs
    ) -> Dict[str, Any]:
        """
        Run complete CA → Story pipeline.
        
        1. Run hybrid CA simulation
        2. Convert to world_dna
        3. Generate story via LLM
        4. Sync embeddings
        5. Return complete results
        """
        from CALab.hybrid_ca_aggregator import run_hybrid_simulation
        
        print(f"Running CA simulation ({generations} generations)...")
        world_state = run_hybrid_simulation(
            world_size=world_size,
            generations=generations,
            density=density,
            seed=seed
        )
        
        print("Creating story context...")
        context = self.load_world_state(world_state)
        
        print("Syncing embeddings...")
        self.sync_embeddings_to_db(context)
        
        print("Generating story...")
        story, critique, extra = self.generate_story(context, **story_kwargs)
        
        return {
            'world_state': world_state,
            'world_dna': context.world_dna,
            'entities': context.entities,
            'relationships': context.relationships,
            'story': story,
            'critique': critique,
            'extra': extra,
            'metadata': context.generation_metadata
        }


def quick_ca_story(
    generations: int = 30,
    world_size: Tuple[int, int] = (60, 60),
    theme: str = "",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to generate a story from CA simulation.
    """
    adapter = CAAdapter()
    return adapter.run_full_pipeline(
        generations=generations,
        world_size=world_size,
        theme=theme,
        **kwargs
    )


if __name__ == "__main__":
    print("CA Adapter for StoryLab - Integration Test")
    print("=" * 50)
    
    from CALab.hybrid_ca_aggregator import run_hybrid_simulation
    
    print("\nStep 1: Running quick CA simulation...")
    world_state = run_hybrid_simulation(
        world_size=(50, 50),
        generations=20,
        density=0.3,
        seed=42
    )
    
    print(f"\n  Generated {len(world_state.species)} species")
    print(f"  Generated {len(world_state.characters)} characters")
    print(f"  Generated {len(world_state.locations)} locations")
    
    print("\nStep 2: Creating adapter and context...")
    adapter = CAAdapter()
    context = adapter.load_world_state(world_state)
    
    print(f"\n  World DNA length: {len(context.world_dna)} chars")
    print(f"  Entities extracted: {sum(len(v) for v in context.entities.values())}")
    
    print("\nStep 3: World DNA Preview:")
    print("-" * 40)
    print(context.world_dna[:800])
    print("-" * 40)
    
    print("\nStep 4: Generating story (skipped in test mode)")
    print("To generate story, call:")
    print("  story, critique, extra = adapter.generate_story(context)")
    
    print("\nAdapter ready for StoryLab integration!")
