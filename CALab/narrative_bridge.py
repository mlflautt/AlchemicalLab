"""
Narrative Bridge - Converts CA WorldState to StoryLab world_dna format
=======================================================================

Transforms outputs from CALab systems into StoryLab-compatible world_dna
Markdown format for LLM-driven narrative generation.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import os


class WorldDNAGenerator:
    """
    Generates world_dna.md from CA-derived WorldState.
    """
    
    def __init__(self):
        self.template_sections = [
            'metadata', 'physical_parameters', 'species_parameters',
            'cultural_parameters', 'narrative_parameters', 'llm_parameters',
            'simulation_parameters', 'core_setting', 'key_themes',
            'major_factions', 'protagonist_archetype', 'conflict_seeds',
            'narrative_hooks'
        ]
    
    def generate_world_dna(
        self,
        world_state,
        world_name: str = "Emergent World",
        genre: str = "Fantasy",
        model: str = "DeepSeek-R1",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate complete world_dna.md content from WorldState.
        """
        sections = []
        
        sections.append(self._generate_header())
        sections.append(self._generate_metadata(world_state, world_name))
        sections.append(self._generate_physical_parameters(world_state))
        sections.append(self._generate_species_parameters(world_state))
        sections.append(self._generate_cultural_parameters(world_state))
        sections.append(self._generate_narrative_parameters(world_state))
        sections.append(self._generate_llm_parameters(model, temperature, max_tokens))
        sections.append(self._generate_simulation_parameters(world_state))
        sections.append(self._generate_core_setting(world_state, genre))
        sections.append(self._generate_key_themes(world_state))
        sections.append(self._generate_major_factions(world_state))
        sections.append(self._generate_protagonist_archetype(world_state))
        sections.append(self._generate_conflict_seeds(world_state))
        sections.append(self._generate_narrative_hooks(world_state))
        
        return "\n\n".join(sections)
    
    def _generate_header(self) -> str:
        return "# World-DNA: Seed for Emergent Narrative"
    
    def _generate_metadata(self, world_state, world_name: str) -> str:
        num_species = len(world_state.species)
        num_factions = len(world_state.factions)
        num_chars = len(world_state.characters)
        
        tags = ["emergent", "ca-generated"]
        if num_species > 5:
            tags.append("complex-ecosystem")
        if num_factions > 2:
            tags.append("political")
        if any(s.species_type == 'apex_predator' for s in world_state.species.values()):
            tags.append("dangerous")
        
        return f"""## Metadata
- **Name**: {world_name}
- **Version**: 1.0
- **Description**: An emergent world with {num_species} species, {num_factions} factions, and {num_chars} notable characters
- **Tags**: {', '.join(tags)}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **CA Generation**: {world_state.generation}"""
    
    def _generate_physical_parameters(self, world_state) -> str:
        h, w = world_state.world_size
        
        biome_list = list(world_state.biomes.keys()) if world_state.biomes else ['temperate', 'mountain', 'desert']
        biome_str = ', '.join(b.title() for b in biome_list[:5])
        
        terrain_diversity = len(world_state.biomes) / 6.0 if world_state.biomes else 0.5
        terrain_diversity = min(1.0, terrain_diversity)
        
        water_coverage = 0.15 if 'wetland' in world_state.biomes else 0.1
        
        return f"""## Physical Parameters
- **World Size**: {h}x{w} units
- **Terrain Diversity**: {terrain_diversity:.1f}
- **Climate Stability**: Moderate (0.6)
- **Primary Biomes**: {biome_str}
- **Water Coverage**: {water_coverage:.0%}
- **Natural Disaster Frequency**: Moderate (0.4)"""
    
    def _generate_species_parameters(self, world_state) -> str:
        num_species = len(world_state.species)
        
        species_types = {}
        for s in world_state.species.values():
            species_types[s.species_type] = species_types.get(s.species_type, 0) + 1
        
        interaction_complexity = "High" if len(world_state.ecological_relationships) > 10 else "Moderate"
        migration = sum(s.traits.get('speed', 0.5) for s in world_state.species.values()) / max(1, num_species)
        
        return f"""## Species Parameters
- **Initial Species Count**: {min(4, num_species)}
- **Max Species Count**: {max(10, num_species + 5)}
- **Interaction Complexity**: {interaction_complexity}
- **Behavioral Traits**: 8
- **Cultural Evolution**: Enabled
- **Migration Tendency**: {migration:.1f}
- **Species Distribution**: {dict(species_types)}"""
    
    def _generate_cultural_parameters(self, world_state) -> str:
        num_factions = len(world_state.factions)
        num_chars = len(world_state.characters)
        
        trade_density = len(world_state.narrative_relationships) / max(1, num_chars + num_factions) if (num_chars + num_factions) > 0 else 0.5
        
        return f"""## Cultural Parameters
- **Technology Progression**: Moderate (0.06)
- **Cultural Exchange**: High (0.7)
- **Language Diversity**: {max(2, num_factions)}
- **Religious Complexity**: {'High' if num_factions > 2 else 'Moderate'}
- **Trade Network Density**: {trade_density:.1f}
- **Art Emphasis**: High (0.8)"""
    
    def _generate_narrative_parameters(self, world_state) -> str:
        num_arcs = len(world_state.story_arcs)
        
        tension = 0.5
        for arc in world_state.story_arcs.values():
            tension = max(tension, arc.tension_level)
        
        return f"""## Narrative Parameters
- **Story Complexity**: {'High' if num_arcs > 3 else 'Moderate'}
- **Character Development**: High
- **Historical Detail**: High
- **World History Depth**: {world_state.generation} generations
- **Dramatic Tension**: {tension:.1f}
- **Mythological Elements**: Moderate (0.5)"""
    
    def _generate_llm_parameters(self, model: str, temperature: float, max_tokens: int) -> str:
        return f"""## LLM Parameters
- **Model**: {model}
- **Temperature**: {temperature}
- **Max Tokens**: {max_tokens}
- **Enhancement Frequency**: 0.8
- **Enhancement Types**: species_names, cultural_descriptions, character_personalities, location_descriptions, historical_events"""
    
    def _generate_simulation_parameters(self, world_state) -> str:
        return f"""## Simulation Parameters
- **Total Generations**: {world_state.generation + 100}
- **Steps per Generation**: 10
- **Analysis Frequency**: 25
- **Output Detail**: High"""
    
    def _generate_core_setting(self, world_state, genre: str) -> str:
        locations = list(world_state.locations.values())
        
        if locations:
            primary_location = locations[0].name
            loc_desc = locations[0].atmosphere if hasattr(locations[0], 'atmosphere') else 'mysterious'
        else:
            primary_location = "The Emergent Lands"
            loc_desc = "diverse"
        
        dominant_species = None
        if world_state.species:
            dominant_species = max(
                world_state.species.values(),
                key=lambda s: s.population
            )
        
        setting_desc = f"A world shaped by emergent evolution"
        if dominant_species:
            setting_desc += f", dominated by {dominant_species.species_type}s like the {dominant_species.name}"
        
        return f"""## Core Setting
- **Genre**: {genre}
- **Primary Location**: {primary_location}, a {loc_desc} region
- **Time Period**: An age of transformation and discovery
- **Description**: {setting_desc}"""
    
    def _generate_key_themes(self, world_state) -> str:
        themes = []
        
        if any(s.species_type == 'apex_predator' for s in world_state.species.values()):
            themes.append("- Survival and adaptation in a dangerous world")
        
        if len(world_state.factions) > 1:
            themes.append("- Political intrigue and factional conflict")
        
        if len(world_state.story_arcs) > 0:
            arc_types = set(arc.arc_type for arc in world_state.story_arcs.values())
            if 'mystery' in arc_types:
                themes.append("- Uncovering hidden truths and ancient secrets")
            if 'conflict' in arc_types:
                themes.append("- War and its consequences")
            if 'quest' in arc_types:
                themes.append("- Journeys of discovery and transformation")
        
        if world_state.biomes:
            themes.append("- Balance between civilization and nature")
        
        if not themes:
            themes = [
                "- Discovery and exploration",
                "- Emergence of new order from chaos",
                "- Connections forming across boundaries"
            ]
        
        return "## Key Themes\n" + "\n".join(themes[:5])
    
    def _generate_major_factions(self, world_state) -> str:
        factions_text = []
        
        for faction in list(world_state.factions.values())[:5]:
            ideology = faction.ideology if hasattr(faction, 'ideology') else 'unknown'
            goal = faction.goal if hasattr(faction, 'goal') else 'unknown'
            desc = f"**{faction.name}**: Believes in {ideology.lower()}, seeks to {goal.lower()}."
            factions_text.append(f"- {desc}")
        
        if not factions_text:
            for species in list(world_state.species.values())[:3]:
                if species.species_type in ['carnivore', 'apex_predator', 'omnivore']:
                    faction_name = f"The {species.name.split()[0]} Clan"
                    factions_text.append(f"- **{faction_name}**: A group known for their {species.species_type} nature.")
        
        if not factions_text:
            factions_text = [
                "- **The First**: The original inhabitants seeking stability.",
                "- **The Changed**: New arrivals bringing transformation.",
                "- **The Watchers**: Mysterious observers of the emergent world."
            ]
        
        return "## Major Factions\n" + "\n".join(factions_text)
    
    def _generate_protagonist_archetype(self, world_state) -> str:
        characters = list(world_state.characters.values())
        
        if characters:
            char = characters[0]
            char_class = char.properties.get('class', 'wanderer')
            motivation = char.properties.get('motivation', 'discovery')
            flaw = char.properties.get('flaw', 'uncertainty')
            
            return f"""## Protagonist Archetype
- A {char_class.lower()} driven by {motivation.lower()}, struggling with {flaw.lower()}.
- Named **{char.name}**, connected to {len(char.relationships)} other figures."""
        
        if world_state.species:
            species = list(world_state.species.values())[0]
            return f"""## Protagonist Archetype
- A member of the {species.name}, seeking to understand their place in the world.
- With a population of {species.population}, they represent the {species.species_type} archetype."""
        
        return """## Protagonist Archetype
- An emergent being, newly conscious of the world around them.
- Connected to the patterns that shaped their existence."""
    
    def _generate_conflict_seeds(self, world_state) -> str:
        conflicts = []
        
        for rel in world_state.ecological_relationships[:3]:
            if rel.relationship_type == 'predation':
                pred = world_state.species.get(rel.predator_id)
                prey = world_state.species.get(rel.prey_id)
                if pred and prey:
                    conflicts.append(f"- The eternal struggle between {pred.name} and {prey.name}.")
            elif rel.relationship_type == 'competition':
                conflicts.append("- Resources grow scarce, forcing difficult choices.")
        
        for arc in world_state.story_arcs.values():
            if arc.arc_type == 'conflict':
                conflicts.append(f"- Rising tensions threaten to engulf the region.")
            elif arc.arc_type == 'mystery':
                conflicts.append(f"- Hidden dangers lurk beneath the surface.")
        
        if len(world_state.species) > 5:
            conflicts.append("- An overpopulated world strains under the pressure.")
        
        if not conflicts:
            conflicts = [
                "- Resources become scarce as populations grow.",
                "- Old alliances fracture under new pressures.",
                "- Unknown forces shape events from the shadows."
            ]
        
        return "## Conflict Seeds\n" + "\n".join(conflicts[:5])
    
    def _generate_narrative_hooks(self, world_state) -> str:
        hooks = []
        
        for loc in list(world_state.locations.values())[:3]:
            hooks.append(f"- **{loc.name}**: A place of {loc.atmosphere} atmosphere, hiding secrets.")
        
        for species in list(world_state.species.values())[:2]:
            if species.species_type in ['apex_predator', 'carnivore']:
                hooks.append(f"- The {species.name} hunts in unexpected patterns.")
        
        for arc in list(world_state.story_arcs.values())[:2]:
            if arc.story_beats:
                hooks.append(f"- {arc.story_beats[0]}")
        
        if not hooks:
            hooks = [
                "- Ancient patterns suggest deeper meaning.",
                "- Connections form between unlikely allies.",
                "- The world itself seems to guide events."
            ]
        
        return "## Narrative Hooks\n" + "\n".join(hooks[:5])
    
    def save_world_dna(self, world_state, output_path: str, **kwargs) -> str:
        """
        Generate and save world_dna.md to file.
        """
        content = self.generate_world_dna(world_state, **kwargs)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        return content


def world_state_to_world_dna(
    world_state,
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to convert WorldState to world_dna format.
    
    Args:
        world_state: WorldState object from hybrid aggregator
        output_path: Optional path to save the generated markdown
        **kwargs: Additional parameters for generation (world_name, genre, etc.)
    
    Returns:
        The generated world_dna markdown content
    """
    generator = WorldDNAGenerator()
    
    if output_path:
        return generator.save_world_dna(world_state, output_path, **kwargs)
    else:
        return generator.generate_world_dna(world_state, **kwargs)


def generate_story_elements_from_species(species_data) -> List[Dict[str, Any]]:
    """
    Convert species data into narrative story elements.
    """
    elements = []
    
    for species in species_data.values():
        element = {
            'source_type': 'species',
            'source_id': species.species_id,
            'name': species.name,
            'element_type': _species_to_element_type(species.species_type),
            'properties': {
                'species_type': species.species_type,
                'population': species.population,
                'fitness': species.fitness,
                'traits': species.traits,
                'biomes': species.preferred_biomes
            },
            'narrative_hooks': _generate_species_hooks(species)
        }
        elements.append(element)
    
    return elements


def _species_to_element_type(species_type: str) -> str:
    """Map species types to narrative element types."""
    mapping = {
        'apex_predator': 'character',
        'carnivore': 'character',
        'herbivore': 'character',
        'omnivore': 'character',
        'producer': 'location',
        'decomposer': 'mystery'
    }
    return mapping.get(species_type, 'character')


def _generate_species_hooks(species) -> List[str]:
    """Generate narrative hooks for a species."""
    hooks = []
    
    if species.species_type == 'apex_predator':
        hooks.append(f"The {species.name} dominates its territory with ruthless efficiency.")
    elif species.species_type == 'carnivore':
        hooks.append(f"Hunters speak of the {species.name} with fear and respect.")
    elif species.species_type == 'producer':
        hooks.append(f"The {species.name} forms the foundation of life in this region.")
    
    if species.population > 200:
        hooks.append(f"A massive population of {species.name} strains local resources.")
    elif species.population < 50:
        hooks.append(f"The {species.name} faces extinction - only {species.population} remain.")
    
    return hooks


if __name__ == "__main__":
    from CALab.hybrid_ca_aggregator import run_hybrid_simulation, WorldState
    
    print("Narrative Bridge - WorldState to World-DNA Conversion Test")
    print("=" * 55)
    
    print("\nRunning hybrid simulation...")
    world_state = run_hybrid_simulation(
        world_size=(60, 60),
        generations=30,
        density=0.3,
        seed=42
    )
    
    print("\nGenerating world_dna.md...")
    content = world_state_to_world_dna(
        world_state,
        world_name="Test Emergent World",
        genre="Fantasy"
    )
    
    print("\n" + "=" * 55)
    print(content)
    print("=" * 55)
    
    output_path = "/home/mitchellflautt/AlchemicalLab/StoryLab/generated_world_dna.md"
    print(f"\nSaving to {output_path}...")
    world_state_to_world_dna(world_state, output_path, world_name="Generated CA World")
    print("Done!")
