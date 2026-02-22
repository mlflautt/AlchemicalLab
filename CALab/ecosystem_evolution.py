"""
Ecosystem Evolution Simulation using CA-Graph Emergence
Species emerge from CA patterns, evolve traits, and form complex ecological networks
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Polygon
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import time
from datetime import datetime

plt.style.use('dark_background')

# Species archetypes
SPECIES_TYPES = {
    'producer': {'color': '#2ed573', 'symbol': '♦', 'trophic_level': 1, 'energy_source': 'photosynthesis'},
    'herbivore': {'color': '#feca57', 'symbol': '●', 'trophic_level': 2, 'energy_source': 'plants'},
    'carnivore': {'color': '#ff6b6b', 'symbol': '▲', 'trophic_level': 3, 'energy_source': 'prey'},
    'omnivore': {'color': '#ff9ff3', 'symbol': '◆', 'trophic_level': 2.5, 'energy_source': 'mixed'},
    'decomposer': {'color': '#5f27cd', 'symbol': '■', 'trophic_level': 1, 'energy_source': 'detritus'},
    'apex_predator': {'color': '#ff4757', 'symbol': '★', 'trophic_level': 4, 'energy_source': 'apex_hunting'}
}

# Environmental zones
BIOME_TYPES = {
    'forest': {'productivity': 0.8, 'stability': 0.9, 'color': '#2d8e2d'},
    'grassland': {'productivity': 0.7, 'stability': 0.7, 'color': '#7cb342'},
    'desert': {'productivity': 0.3, 'stability': 0.8, 'color': '#d4af37'},
    'wetland': {'productivity': 0.9, 'stability': 0.6, 'color': '#4fb3d9'},
    'mountain': {'productivity': 0.4, 'stability': 0.9, 'color': '#5d6d7e'},
    'tundra': {'productivity': 0.2, 'stability': 0.8, 'color': '#85929e'}
}

@dataclass
class Species:
    """A species that evolved from CA patterns."""
    species_id: int
    species_name: str
    species_type: str
    center: Tuple[int, int]
    birth_generation: int
    last_seen: int
    
    # Evolutionary traits
    traits: Dict[str, float] = field(default_factory=dict)
    population: int = 100
    territory_size: int = 5
    
    # Ecological properties
    energy_efficiency: float = 0.5
    reproduction_rate: float = 0.1
    mutation_rate: float = 0.01
    dispersal_range: int = 10
    
    # Environmental preferences
    preferred_biomes: List[str] = field(default_factory=list)
    temperature_tolerance: Tuple[float, float] = (0.0, 1.0)
    
    # Dynamic state
    fitness: float = 1.0
    stress_level: float = 0.0
    generation_alive: int = 0

@dataclass
class EcologicalRelationship:
    """Relationship between species in the ecosystem."""
    predator_id: int
    prey_id: int
    relationship_type: str  # predation, competition, mutualism, parasitism
    strength: float = 1.0
    stability: float = 0.5
    coevolution_pressure: float = 0.1
    
    # Interaction history
    interaction_events: List[str] = field(default_factory=list)
    age: int = 0

@dataclass
class EnvironmentalZone:
    """Environmental zone with specific characteristics."""
    zone_id: int
    biome_type: str
    center: Tuple[int, int]
    radius: int
    
    # Environmental parameters
    temperature: float = 0.5
    moisture: float = 0.5
    nutrients: float = 0.5
    disturbance_level: float = 0.1
    
    # Dynamic properties
    carrying_capacity: int = 1000
    current_biomass: float = 0.0
    biodiversity_index: float = 0.0

class SpeciesNameGenerator:
    """Generates scientifically-inspired species names."""
    
    def __init__(self):
        self.genus_prefixes = [
            "Neo", "Proto", "Pseudo", "Meta", "Para", "Super", "Ultra", "Macro", "Micro", 
            "Hyper", "Sub", "Trans", "Inter", "Extra", "Multi", "Omni", "Poly", "Mono"
        ]
        
        self.genus_roots = [
            "thymus", "phyton", "zoon", "morph", "trophic", "genetic", "biotic", "cellular",
            "neural", "cardio", "gastro", "hepato", "nephro", "pulmo", "osseo", "muscular"
        ]
        
        self.species_suffixes = [
            "ensis", "icus", "atus", "osis", "ensis", "alis", "aris", "eus", "oides", 
            "iformis", "philus", "phagus", "vorus", "genesis", "tropic", "morphic"
        ]
    
    def generate_species_name(self, species_type: str, traits: Dict[str, float]) -> str:
        """Generate a species name based on type and traits."""
        # Choose components based on species characteristics
        prefix = random.choice(self.genus_prefixes)
        root = random.choice(self.genus_roots)
        suffix = random.choice(self.species_suffixes)
        
        # Modify based on species type
        if species_type == 'apex_predator':
            prefix = random.choice(["Mega", "Ultra", "Super", "Hyper"])
        elif species_type == 'producer':
            root = random.choice(["phyton", "chloro", "photo", "auto"])
        elif species_type == 'decomposer':
            root = random.choice(["sapro", "detrito", "necro", "lyso"])
        
        genus = f"{prefix}{root}"
        species = f"{genus.lower()} {suffix}"
        
        return species.title()

class EcosystemSimulation:
    """CA-driven ecosystem evolution simulation."""
    
    def __init__(self, world_size: Tuple[int, int] = (150, 150)):
        self.world_size = world_size
        self.ca_grid = np.zeros(world_size, dtype=bool)
        self.generation = 0
        
        # Species and ecosystem components
        self.species: Dict[int, Species] = {}
        self.relationships: Dict[Tuple[int, int], EcologicalRelationship] = {}
        self.environmental_zones: Dict[int, EnvironmentalZone] = {}
        
        self.species_id_counter = 0
        self.zone_id_counter = 0
        self.name_generator = SpeciesNameGenerator()
        
        # Pattern detection
        from emergent_graphs import CAPatternDetector
        self.detector = CAPatternDetector()
        
        # Simulation parameters
        self.speciation_threshold = 12  # Pattern size for new species
        self.extinction_threshold = 5   # Population below which species go extinct
        self.interaction_range = 20.0   # Range for ecological interactions
        
        # Environmental dynamics
        self.climate_change_rate = 0.001
        self.disturbance_frequency = 0.05
        self.carrying_capacity_pressure = 0.8
        
        # Evolution tracking
        self.evolution_history: List[Dict] = []
        self.extinction_events: List[Dict] = []
        self.speciation_events: List[Dict] = []
        
        # File naming for saves
        self.simulation_start_time = datetime.now()
        
    def initialize_ecosystem(self, density: float = 0.3, seed: int = 42):
        """Initialize ecosystem with environmental zones."""
        np.random.seed(seed)
        random.seed(seed)
        
        h, w = self.world_size
        
        # Create diverse biomes
        biome_centers = [
            (h//4, w//4, 'forest'),
            (h//4, 3*w//4, 'grassland'),
            (3*h//4, w//4, 'wetland'),
            (3*h//4, 3*w//4, 'desert'),
            (h//2, w//2, 'mountain'),
            (h//8, w//2, 'tundra')
        ]
        
        for center_y, center_x, biome_type in biome_centers:
            zone_id = self.zone_id_counter
            self.zone_id_counter += 1
            
            radius = random.randint(15, 30)
            
            zone = EnvironmentalZone(
                zone_id=zone_id,
                biome_type=biome_type,
                center=(center_y, center_x),
                radius=radius,
                temperature=random.uniform(0.2, 0.8),
                moisture=random.uniform(0.3, 0.9),
                nutrients=random.uniform(0.4, 0.8)
            )
            
            self.environmental_zones[zone_id] = zone
            
            # Seed CA patterns in biome
            self._seed_biome_patterns(center_y, center_x, radius, biome_type, density)
    
    def _seed_biome_patterns(self, center_y: int, center_x: int, radius: int, biome_type: str, density: float):
        """Seed CA patterns specific to biome type."""
        h, w = self.world_size
        
        # Create biome mask
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        biome_mask = distances <= radius
        
        # Biome-specific seeding patterns
        biome_density = density * BIOME_TYPES[biome_type]['productivity']
        
        if biome_type == 'forest':
            # Dense patches with clearings
            for _ in range(random.randint(3, 6)):
                patch_y = center_y + random.randint(-radius//2, radius//2)
                patch_x = center_x + random.randint(-radius//2, radius//2)
                patch_radius = random.randint(3, 8)
                
                patch_mask = (y_coords - patch_y)**2 + (x_coords - patch_x)**2 <= patch_radius**2
                combined_mask = biome_mask & patch_mask
                self.ca_grid[combined_mask] |= np.random.random(np.sum(combined_mask)) < biome_density * 1.5
                
        elif biome_type == 'grassland':
            # Scattered patterns
            self.ca_grid[biome_mask] |= np.random.random(np.sum(biome_mask)) < biome_density
            
        elif biome_type == 'wetland':
            # Linear features (rivers, streams)
            for _ in range(random.randint(2, 4)):
                # Create winding path
                start_y, start_x = center_y + random.randint(-radius, radius), center_x + random.randint(-radius, radius)
                for step in range(20):
                    if 0 <= start_y < h and 0 <= start_x < w and distances[start_y, start_x] <= radius:
                        self.ca_grid[start_y-1:start_y+2, start_x-1:start_x+2] = True
                    start_y += random.randint(-2, 2)
                    start_x += random.randint(-2, 2)
        
        elif biome_type == 'desert':
            # Sparse, clustered patterns around oases
            oases_count = random.randint(1, 3)
            for _ in range(oases_count):
                oasis_y = center_y + random.randint(-radius//3, radius//3)
                oasis_x = center_x + random.randint(-radius//3, radius//3)
                oasis_radius = random.randint(2, 5)
                
                oasis_mask = (y_coords - oasis_y)**2 + (x_coords - oasis_x)**2 <= oasis_radius**2
                combined_mask = biome_mask & oasis_mask
                self.ca_grid[combined_mask] |= np.random.random(np.sum(combined_mask)) < biome_density * 2.0
        
        elif biome_type == 'mountain':
            # Elevation-based gradients
            elevation_gradient = np.exp(-distances / (radius * 0.7))
            mountain_mask = biome_mask & (elevation_gradient > 0.3)
            self.ca_grid[mountain_mask] |= np.random.random(np.sum(mountain_mask)) < biome_density * 0.8
        
        elif biome_type == 'tundra':
            # Patchy, sparse patterns
            self.ca_grid[biome_mask] |= np.random.random(np.sum(biome_mask)) < biome_density * 0.5
    
    def step_ecosystem(self):
        """Step the ecosystem simulation."""
        # 1. CA evolution with ecological influences
        self._step_ca_with_ecology()
        
        # 2. Species detection and speciation
        if self.generation % 4 == 0:
            self._detect_and_evolve_species()
        
        # 3. Ecological interactions
        if self.generation % 3 == 0:
            self._evolve_ecological_relationships()
        
        # 4. Population dynamics
        if self.generation % 2 == 0:
            self._update_population_dynamics()
        
        # 5. Environmental changes
        if self.generation % 10 == 0:
            self._apply_environmental_changes()
        
        # 6. Record evolution
        if self.generation % 5 == 0:
            self._record_evolutionary_state()
        
        self.generation += 1
    
    def _step_ca_with_ecology(self):
        """Step CA with ecological influences."""
        from scipy import ndimage
        
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        padded = np.pad(self.ca_grid.astype(int), 1, mode='wrap')
        neighbors = ndimage.convolve(padded, kernel, mode='constant')[1:-1, 1:-1]
        
        # Base Conway rules
        new_grid = np.zeros_like(self.ca_grid)
        new_grid |= (~self.ca_grid) & (neighbors == 3)
        new_grid |= self.ca_grid & ((neighbors == 2) | (neighbors == 3))
        
        # Apply species influences
        self._apply_species_influences(new_grid, neighbors)
        
        # Apply environmental influences
        self._apply_environmental_influences(new_grid)
        
        self.ca_grid = new_grid
    
    def _apply_species_influences(self, new_grid: np.ndarray, neighbors: np.ndarray):
        """Apply species-based influences to CA evolution."""
        for species in self.species.values():
            y, x = species.center
            influence_radius = species.territory_size + species.dispersal_range // 3
            
            y_min = max(0, y - influence_radius)
            y_max = min(self.world_size[0], y + influence_radius + 1)
            x_min = max(0, x - influence_radius)  
            x_max = min(self.world_size[1], x + influence_radius + 1)
            
            influence_strength = species.population / 1000.0 * species.energy_efficiency
            
            if species.species_type == 'producer':
                # Producers spread and stabilize patterns
                growth_chance = min(0.3, influence_strength * 0.5)
                growth_mask = np.random.random((y_max-y_min, x_max-x_min)) < growth_chance
                new_grid[y_min:y_max, x_min:x_max] |= growth_mask
                
            elif species.species_type in ['herbivore', 'omnivore']:
                # Herbivores create clearings by consuming
                if influence_strength > 0.1:
                    consumption_chance = min(0.2, influence_strength * 0.3)
                    consumption_mask = np.random.random((y_max-y_min, x_max-x_min)) < consumption_chance
                    new_grid[y_min:y_max, x_min:x_max] &= ~consumption_mask
                    
            elif species.species_type in ['carnivore', 'apex_predator']:
                # Predators indirectly affect through prey pressure
                # Create dynamic hunting patterns
                if species.stress_level < 0.5:  # Well-fed predators create stable territories
                    stability_chance = influence_strength * 0.1
                    stability_mask = np.random.random((y_max-y_min, x_max-x_min)) < stability_chance
                    current_region = new_grid[y_min:y_max, x_min:x_max]
                    new_grid[y_min:y_max, x_min:x_max] = current_region | stability_mask
                    
            elif species.species_type == 'decomposer':
                # Decomposers recycle nutrients, promoting growth nearby
                recycling_chance = min(0.25, influence_strength * 0.4)
                recycling_mask = np.random.random((y_max-y_min, x_max-x_min)) < recycling_chance
                new_grid[y_min:y_max, x_min:x_max] |= recycling_mask
    
    def _apply_environmental_influences(self, new_grid: np.ndarray):
        """Apply environmental zone influences to CA."""
        for zone in self.environmental_zones.values():
            center_y, center_x = zone.center
            radius = zone.radius
            
            # Create zone mask
            y_coords, x_coords = np.ogrid[:self.world_size[0], :self.world_size[1]]
            distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            zone_mask = distances <= radius
            
            # Apply biome-specific effects
            biome_info = BIOME_TYPES[zone.biome_type]
            productivity = biome_info['productivity'] * (1 - zone.disturbance_level)
            
            # Environmental influence on pattern formation
            if zone.biome_type in ['forest', 'wetland']:
                # High productivity areas promote growth
                if productivity > 0.6:
                    growth_boost = np.random.random(zone_mask.shape) < (productivity - 0.6) * 0.2
                    new_grid[zone_mask] |= growth_boost[zone_mask]
                    
            elif zone.biome_type in ['desert', 'tundra']:
                # Low productivity areas limit growth
                if productivity < 0.4:
                    limitation = np.random.random(zone_mask.shape) < (0.4 - productivity) * 0.15
                    new_grid[zone_mask] &= ~limitation[zone_mask]
    
    def _detect_and_evolve_species(self):
        """Detect new species from CA patterns and evolve existing ones."""
        current_patterns = self.detector.detect_patterns(self.ca_grid, self.generation)
        
        for pattern in current_patterns:
            if pattern.size >= self.speciation_threshold:
                if not self._matches_existing_species(pattern):
                    new_species = self._create_new_species(pattern)
                    if new_species:
                        self.species[new_species.species_id] = new_species
                        self._record_speciation_event(new_species)
        
        # Evolve existing species
        self._evolve_existing_species()
    
    def _matches_existing_species(self, pattern) -> bool:
        """Check if pattern matches an existing species territory."""
        for species in self.species.values():
            distance = np.sqrt((pattern.center[0] - species.center[0])**2 + 
                             (pattern.center[1] - species.center[1])**2)
            
            territory_range = species.territory_size + species.dispersal_range
            if distance <= territory_range:
                # Update existing species
                species.last_seen = self.generation
                
                # Possible migration or territory expansion
                if distance > species.territory_size:
                    species.center = pattern.center  # Migration
                    species.dispersal_range = min(species.dispersal_range + 1, 25)
                
                return True
        return False
    
    def _create_new_species(self, pattern) -> Optional[Species]:
        """Create a new species from a CA pattern."""
        # Determine species type based on environmental context
        species_type = self._determine_species_type(pattern)
        
        species_id = self.species_id_counter
        self.species_id_counter += 1
        
        # Generate traits based on pattern and environment
        traits = self._generate_species_traits(pattern, species_type)
        
        # Generate name
        species_name = self.name_generator.generate_species_name(species_type, traits)
        
        new_species = Species(
            species_id=species_id,
            species_name=species_name,
            species_type=species_type,
            center=pattern.center,
            birth_generation=self.generation,
            last_seen=self.generation,
            traits=traits,
            population=random.randint(50, 200),
            territory_size=max(5, pattern.size),
            energy_efficiency=random.uniform(0.3, 0.8),
            reproduction_rate=random.uniform(0.05, 0.15),
            mutation_rate=random.uniform(0.005, 0.02),
            dispersal_range=random.randint(8, 18)
        )
        
        # Set environmental preferences
        new_species.preferred_biomes = self._determine_biome_preferences(pattern, species_type)
        new_species.temperature_tolerance = self._determine_temperature_tolerance(species_type)
        
        return new_species
    
    def _determine_species_type(self, pattern) -> str:
        """Determine species type based on pattern characteristics and environment."""
        y, x = pattern.center
        
        # Check which environmental zone this pattern is in
        local_biome = None
        min_distance = float('inf')
        
        for zone in self.environmental_zones.values():
            zone_y, zone_x = zone.center
            distance = np.sqrt((y - zone_y)**2 + (x - zone_x)**2)
            if distance <= zone.radius and distance < min_distance:
                min_distance = distance
                local_biome = zone.biome_type
        
        # Bias species type based on biome
        if local_biome == 'forest' or local_biome == 'grassland':
            weights = {'producer': 0.4, 'herbivore': 0.3, 'carnivore': 0.15, 'omnivore': 0.1, 'decomposer': 0.05}
        elif local_biome == 'wetland':
            weights = {'producer': 0.35, 'herbivore': 0.25, 'carnivore': 0.2, 'omnivore': 0.15, 'decomposer': 0.05}
        elif local_biome == 'desert':
            weights = {'producer': 0.25, 'herbivore': 0.2, 'carnivore': 0.25, 'omnivore': 0.25, 'decomposer': 0.05}
        elif local_biome == 'mountain':
            weights = {'producer': 0.2, 'herbivore': 0.3, 'carnivore': 0.3, 'omnivore': 0.15, 'apex_predator': 0.05}
        elif local_biome == 'tundra':
            weights = {'producer': 0.15, 'herbivore': 0.35, 'carnivore': 0.3, 'omnivore': 0.15, 'apex_predator': 0.05}
        else:
            weights = {'producer': 0.25, 'herbivore': 0.25, 'carnivore': 0.25, 'omnivore': 0.2, 'decomposer': 0.05}
        
        # Pattern size influences type
        if pattern.size > 20:
            weights['apex_predator'] = weights.get('apex_predator', 0) + 0.1
            weights['producer'] = max(0, weights['producer'] - 0.1)
        elif pattern.size < 8:
            weights['decomposer'] += 0.1
            weights['carnivore'] = max(0, weights['carnivore'] - 0.1)
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Choose species type
        rand_val = random.random()
        cumulative = 0
        for species_type, weight in weights.items():
            cumulative += weight
            if rand_val <= cumulative:
                return species_type
        
        return 'omnivore'  # Fallback
    
    def _generate_species_traits(self, pattern, species_type: str) -> Dict[str, float]:
        """Generate species-specific traits."""
        traits = {}
        
        # Base traits that all species have
        traits['size'] = random.uniform(0.1, 2.0)
        traits['speed'] = random.uniform(0.2, 1.5)
        traits['intelligence'] = random.uniform(0.1, 1.0)
        traits['social'] = random.uniform(0.0, 1.0)
        traits['aggression'] = random.uniform(0.1, 1.0)
        traits['camouflage'] = random.uniform(0.0, 1.0)
        
        # Type-specific trait modifications
        if species_type == 'producer':
            traits['photosynthesis_efficiency'] = random.uniform(0.6, 1.0)
            traits['root_depth'] = random.uniform(0.3, 1.0)
            traits['drought_tolerance'] = random.uniform(0.2, 0.9)
            traits['size'] = random.uniform(0.5, 3.0)  # Can be large
            
        elif species_type == 'herbivore':
            traits['digestive_efficiency'] = random.uniform(0.5, 0.9)
            traits['foraging_skill'] = random.uniform(0.4, 1.0)
            traits['speed'] = random.uniform(0.6, 1.5)  # Need speed to escape
            traits['group_cohesion'] = random.uniform(0.5, 1.0)
            
        elif species_type == 'carnivore':
            traits['hunting_skill'] = random.uniform(0.6, 1.0)
            traits['bite_force'] = random.uniform(0.5, 1.0)
            traits['stealth'] = random.uniform(0.4, 1.0)
            traits['aggression'] = random.uniform(0.6, 1.0)
            
        elif species_type == 'omnivore':
            traits['dietary_flexibility'] = random.uniform(0.7, 1.0)
            traits['adaptability'] = random.uniform(0.6, 1.0)
            traits['intelligence'] = random.uniform(0.5, 1.0)
            
        elif species_type == 'decomposer':
            traits['decomposition_rate'] = random.uniform(0.5, 1.0)
            traits['toxin_resistance'] = random.uniform(0.6, 1.0)
            traits['size'] = random.uniform(0.05, 0.3)  # Usually small
            
        elif species_type == 'apex_predator':
            traits['hunting_skill'] = random.uniform(0.8, 1.0)
            traits['territory_control'] = random.uniform(0.7, 1.0)
            traits['size'] = random.uniform(1.5, 3.0)  # Large
            traits['intelligence'] = random.uniform(0.6, 1.0)
        
        return traits
    
    def _determine_biome_preferences(self, pattern, species_type: str) -> List[str]:
        """Determine which biomes a species prefers."""
        y, x = pattern.center
        preferences = []
        
        # Find nearby biomes
        for zone in self.environmental_zones.values():
            zone_y, zone_x = zone.center
            distance = np.sqrt((y - zone_y)**2 + (x - zone_x)**2)
            
            if distance <= zone.radius * 1.5:  # Include edge effects
                preferences.append(zone.biome_type)
        
        # If no biomes found, add based on species type
        if not preferences:
            if species_type == 'producer':
                preferences = ['forest', 'grassland', 'wetland']
            elif species_type in ['herbivore', 'omnivore']:
                preferences = ['grassland', 'forest']
            elif species_type in ['carnivore', 'apex_predator']:
                preferences = ['forest', 'mountain', 'tundra']
            else:
                preferences = list(BIOME_TYPES.keys())
        
        return preferences[:3]  # Limit to 3 preferred biomes
    
    def _determine_temperature_tolerance(self, species_type: str) -> Tuple[float, float]:
        """Determine temperature tolerance range."""
        base_range = 0.4
        
        if species_type == 'apex_predator':
            # Apex predators often have broader tolerances
            center = random.uniform(0.3, 0.7)
            range_size = random.uniform(0.5, 0.8)
        elif species_type == 'producer':
            # Producers often specialized to local conditions
            center = random.uniform(0.2, 0.8)
            range_size = random.uniform(0.3, 0.6)
        else:
            center = random.uniform(0.3, 0.7)
            range_size = random.uniform(0.3, 0.7)
        
        min_temp = max(0.0, center - range_size/2)
        max_temp = min(1.0, center + range_size/2)
        
        return (min_temp, max_temp)
    
    def _evolve_existing_species(self):
        """Evolve traits of existing species."""
        species_to_remove = []
        
        for species in self.species.values():
            species.generation_alive += 1
            
            # Check for extinction
            if species.population < self.extinction_threshold:
                species_to_remove.append(species.species_id)
                continue
            
            # Mutate traits
            for trait_name, trait_value in species.traits.items():
                if random.random() < species.mutation_rate:
                    # Gaussian mutation
                    mutation_strength = 0.05
                    new_value = trait_value + random.gauss(0, mutation_strength)
                    species.traits[trait_name] = max(0.0, min(1.0, new_value))
            
            # Evolve other properties
            if random.random() < species.mutation_rate:
                species.energy_efficiency += random.gauss(0, 0.02)
                species.energy_efficiency = max(0.1, min(1.0, species.energy_efficiency))
            
            if random.random() < species.mutation_rate:
                species.reproduction_rate += random.gauss(0, 0.01)
                species.reproduction_rate = max(0.01, min(0.3, species.reproduction_rate))
        
        # Handle extinctions
        for species_id in species_to_remove:
            extinct_species = self.species[species_id]
            self._record_extinction_event(extinct_species)
            del self.species[species_id]
            
            # Remove relationships involving extinct species
            relationships_to_remove = [key for key in self.relationships 
                                     if species_id in key]
            for key in relationships_to_remove:
                del self.relationships[key]
    
    def _evolve_ecological_relationships(self):
        """Evolve predator-prey and other ecological relationships."""
        # Detect new relationships
        self._detect_new_relationships()
        
        # Evolve existing relationships
        relationships_to_remove = []
        for key, relationship in self.relationships.items():
            relationship.age += 1
            
            # Check if both species still exist
            if key[0] not in self.species or key[1] not in self.species:
                relationships_to_remove.append(key)
                continue
            
            # Coevolutionary pressure
            predator = self.species[relationship.predator_id]
            prey = self.species[relationship.prey_id]
            
            if relationship.relationship_type == 'predation':
                # Predator-prey coevolution
                self._apply_coevolutionary_pressure(predator, prey, relationship)
            
            # Relationship strength evolution
            if random.random() < 0.1:  # 10% chance per cycle
                relationship.strength += random.gauss(0, 0.05)
                relationship.strength = max(0.1, min(1.0, relationship.strength))
        
        # Clean up broken relationships
        for key in relationships_to_remove:
            del self.relationships[key]
    
    def _detect_new_relationships(self):
        """Detect new ecological relationships between species."""
        species_list = list(self.species.values())
        
        for i, species1 in enumerate(species_list):
            for species2 in species_list[i+1:]:
                # Skip if relationship already exists
                key1 = (species1.species_id, species2.species_id)
                key2 = (species2.species_id, species1.species_id)
                
                if key1 in self.relationships or key2 in self.relationships:
                    continue
                
                # Check distance
                distance = np.sqrt((species1.center[0] - species2.center[0])**2 + 
                                 (species1.center[1] - species2.center[1])**2)
                
                if distance <= self.interaction_range:
                    relationship_type = self._determine_relationship_type(species1, species2)
                    
                    if relationship_type:
                        # Determine predator/prey roles
                        if relationship_type == 'predation':
                            predator, prey = self._determine_predator_prey(species1, species2)
                            key = (predator.species_id, prey.species_id)
                            
                            new_relationship = EcologicalRelationship(
                                predator_id=predator.species_id,
                                prey_id=prey.species_id,
                                relationship_type=relationship_type,
                                strength=random.uniform(0.3, 0.8),
                                stability=random.uniform(0.4, 0.9),
                                coevolution_pressure=random.uniform(0.05, 0.2)
                            )
                            
                            self.relationships[key] = new_relationship
                        
                        else:
                            # Non-predation relationships
                            key = (min(species1.species_id, species2.species_id), 
                                  max(species1.species_id, species2.species_id))
                            
                            new_relationship = EcologicalRelationship(
                                predator_id=species1.species_id,  # Just first species
                                prey_id=species2.species_id,     # Just second species
                                relationship_type=relationship_type,
                                strength=random.uniform(0.2, 0.7)
                            )
                            
                            self.relationships[key] = new_relationship
    
    def _determine_relationship_type(self, species1: Species, species2: Species) -> Optional[str]:
        """Determine the type of ecological relationship."""
        type1, type2 = species1.species_type, species2.species_type
        
        # Predation relationships
        if type1 == 'carnivore' and type2 == 'herbivore':
            return 'predation'
        elif type1 == 'apex_predator' and type2 in ['carnivore', 'omnivore']:
            return 'predation'
        elif type1 == 'omnivore' and type2 == 'producer':
            return 'predation'
        elif type1 == 'herbivore' and type2 == 'producer':
            return 'predation'
        elif type2 == 'carnivore' and type1 == 'herbivore':
            return 'predation'
        elif type2 == 'apex_predator' and type1 in ['carnivore', 'omnivore']:
            return 'predation'
        elif type2 == 'omnivore' and type1 == 'producer':
            return 'predation'
        elif type2 == 'herbivore' and type1 == 'producer':
            return 'predation'
        
        # Competition (same trophic level)
        elif type1 == type2:
            return 'competition'
        elif SPECIES_TYPES[type1]['trophic_level'] == SPECIES_TYPES[type2]['trophic_level']:
            return 'competition'
        
        # Mutualism (producers and decomposers)
        elif (type1 == 'producer' and type2 == 'decomposer') or (type2 == 'producer' and type1 == 'decomposer'):
            return 'mutualism'
        
        # Default: neutral interaction
        return None
    
    def _determine_predator_prey(self, species1: Species, species2: Species) -> Tuple[Species, Species]:
        """Determine which species is predator and which is prey."""
        trophic1 = SPECIES_TYPES[species1.species_type]['trophic_level']
        trophic2 = SPECIES_TYPES[species2.species_type]['trophic_level']
        
        if trophic1 > trophic2:
            return species1, species2
        elif trophic2 > trophic1:
            return species2, species1
        else:
            # Same trophic level - use size and aggression
            score1 = species1.traits.get('size', 0.5) * species1.traits.get('aggression', 0.5)
            score2 = species2.traits.get('size', 0.5) * species2.traits.get('aggression', 0.5)
            
            if score1 > score2:
                return species1, species2
            else:
                return species2, species1
    
    def _apply_coevolutionary_pressure(self, predator: Species, prey: Species, relationship: EcologicalRelationship):
        """Apply coevolutionary pressure between predator and prey."""
        pressure = relationship.coevolution_pressure
        
        # Predator adaptations
        if 'hunting_skill' in predator.traits:
            predator.traits['hunting_skill'] += random.gauss(0, pressure * 0.5)
            predator.traits['hunting_skill'] = max(0.0, min(1.0, predator.traits['hunting_skill']))
        
        if 'speed' in predator.traits:
            predator.traits['speed'] += random.gauss(0, pressure * 0.3)
            predator.traits['speed'] = max(0.0, min(1.5, predator.traits['speed']))
        
        # Prey adaptations
        if 'speed' in prey.traits:
            prey.traits['speed'] += random.gauss(0, pressure * 0.4)
            prey.traits['speed'] = max(0.0, min(1.5, prey.traits['speed']))
        
        if 'camouflage' in prey.traits:
            prey.traits['camouflage'] += random.gauss(0, pressure * 0.3)
            prey.traits['camouflage'] = max(0.0, min(1.0, prey.traits['camouflage']))
        
        # Group defense for prey
        if 'social' in prey.traits:
            prey.traits['social'] += random.gauss(0, pressure * 0.2)
            prey.traits['social'] = max(0.0, min(1.0, prey.traits['social']))
    
    def _update_population_dynamics(self):
        """Update population sizes based on ecological interactions."""
        for species in self.species.values():
            # Base population change
            growth_rate = species.reproduction_rate - 0.05  # Natural death rate
            
            # Environmental carrying capacity
            local_carrying_capacity = self._calculate_carrying_capacity(species)
            
            # Predation pressure
            predation_pressure = self._calculate_predation_pressure(species)
            
            # Competition pressure
            competition_pressure = self._calculate_competition_pressure(species)
            
            # Calculate population change
            effective_growth_rate = growth_rate - predation_pressure - competition_pressure
            
            # Apply carrying capacity limitations
            if species.population >= local_carrying_capacity:
                effective_growth_rate = min(effective_growth_rate, -0.02)
            
            # Update population
            population_change = int(species.population * effective_growth_rate)
            species.population = max(1, species.population + population_change)
            
            # Update fitness and stress
            species.fitness = self._calculate_fitness(species, local_carrying_capacity)
            species.stress_level = max(0.0, min(1.0, predation_pressure + competition_pressure - 0.1))
    
    def _calculate_carrying_capacity(self, species: Species) -> int:
        """Calculate local carrying capacity for a species."""
        base_capacity = 100
        
        # Find the most suitable biome
        best_suitability = 0.0
        for zone in self.environmental_zones.values():
            if zone.biome_type in species.preferred_biomes:
                zone_y, zone_x = zone.center
                distance = np.sqrt((species.center[0] - zone_y)**2 + (species.center[1] - zone_x)**2)
                
                if distance <= zone.radius:
                    # Temperature suitability
                    temp_suitability = 1.0
                    if not (species.temperature_tolerance[0] <= zone.temperature <= species.temperature_tolerance[1]):
                        temp_diff = min(abs(zone.temperature - species.temperature_tolerance[0]),
                                      abs(zone.temperature - species.temperature_tolerance[1]))
                        temp_suitability = max(0.1, 1.0 - temp_diff * 2)
                    
                    # Biome productivity
                    productivity = BIOME_TYPES[zone.biome_type]['productivity']
                    
                    suitability = temp_suitability * productivity * (1 - zone.disturbance_level)
                    best_suitability = max(best_suitability, suitability)
        
        # Species-specific capacity modifiers
        if species.species_type == 'producer':
            base_capacity = 500
        elif species.species_type in ['herbivore', 'omnivore']:
            base_capacity = 200
        elif species.species_type == 'carnivore':
            base_capacity = 100
        elif species.species_type == 'apex_predator':
            base_capacity = 50
        elif species.species_type == 'decomposer':
            base_capacity = 1000
        
        return int(base_capacity * max(0.1, best_suitability))
    
    def _calculate_predation_pressure(self, species: Species) -> float:
        """Calculate predation pressure on a species."""
        total_pressure = 0.0
        
        for relationship in self.relationships.values():
            if relationship.prey_id == species.species_id and relationship.relationship_type == 'predation':
                predator = self.species.get(relationship.predator_id)
                if predator:
                    # Pressure based on predator population and relationship strength
                    pressure = (predator.population / 1000.0) * relationship.strength * 0.1
                    total_pressure += pressure
        
        return min(0.5, total_pressure)  # Cap at 50% pressure
    
    def _calculate_competition_pressure(self, species: Species) -> float:
        """Calculate competition pressure on a species."""
        total_pressure = 0.0
        
        for relationship in self.relationships.values():
            if relationship.relationship_type == 'competition':
                competitor_id = None
                if relationship.predator_id == species.species_id:
                    competitor_id = relationship.prey_id
                elif relationship.prey_id == species.species_id:
                    competitor_id = relationship.predator_id
                
                if competitor_id:
                    competitor = self.species.get(competitor_id)
                    if competitor:
                        # Competition based on population overlap and similarity
                        pressure = (competitor.population / 1000.0) * relationship.strength * 0.05
                        total_pressure += pressure
        
        return min(0.3, total_pressure)  # Cap at 30% pressure
    
    def _calculate_fitness(self, species: Species, carrying_capacity: int) -> float:
        """Calculate overall fitness of a species."""
        # Population health
        pop_ratio = species.population / max(1, carrying_capacity)
        pop_fitness = 1.0 - abs(pop_ratio - 0.8)  # Optimal at 80% of carrying capacity
        
        # Environmental suitability (already factored into carrying capacity)
        env_fitness = 0.8  # Assume decent environmental fit if species exists
        
        # Genetic diversity (inversely related to inbreeding)
        genetic_fitness = max(0.2, 1.0 - (1.0 / max(10, species.population)) * 100)
        
        # Stress impact
        stress_fitness = 1.0 - species.stress_level
        
        # Age penalty for very old species (encourages turnover)
        age_penalty = max(0.0, (species.generation_alive - 100) / 1000.0)
        
        fitness = (pop_fitness + env_fitness + genetic_fitness + stress_fitness) / 4.0 - age_penalty
        
        return max(0.1, min(1.0, fitness))
    
    def _apply_environmental_changes(self):
        """Apply environmental changes over time."""
        for zone in self.environmental_zones.values():
            # Climate change
            if random.random() < 0.3:  # 30% chance of temperature change
                zone.temperature += random.gauss(0, self.climate_change_rate)
                zone.temperature = max(0.0, min(1.0, zone.temperature))
            
            if random.random() < 0.25:  # 25% chance of moisture change
                zone.moisture += random.gauss(0, self.climate_change_rate)
                zone.moisture = max(0.0, min(1.0, zone.moisture))
            
            # Random disturbances
            if random.random() < self.disturbance_frequency:
                disturbance_strength = random.uniform(0.1, 0.5)
                zone.disturbance_level = min(1.0, zone.disturbance_level + disturbance_strength)
            else:
                # Recovery from disturbance
                zone.disturbance_level = max(0.0, zone.disturbance_level - 0.05)
            
            # Update biomass
            total_biomass = sum(species.population for species in self.species.values() 
                              if self._species_in_zone(species, zone))
            zone.current_biomass = total_biomass
            
            # Update biodiversity
            species_in_zone = sum(1 for species in self.species.values() 
                                if self._species_in_zone(species, zone))
            zone.biodiversity_index = species_in_zone / max(1, len(self.species)) if self.species else 0.0
    
    def _species_in_zone(self, species: Species, zone: EnvironmentalZone) -> bool:
        """Check if a species is within an environmental zone."""
        zone_y, zone_x = zone.center
        species_y, species_x = species.center
        distance = np.sqrt((species_y - zone_y)**2 + (species_x - zone_x)**2)
        return distance <= zone.radius
    
    def _record_evolutionary_state(self):
        """Record current state for evolutionary analysis."""
        state = {
            'generation': self.generation,
            'num_species': len(self.species),
            'total_population': sum(s.population for s in self.species.values()),
            'num_relationships': len(self.relationships),
            'avg_fitness': np.mean([s.fitness for s in self.species.values()]) if self.species else 0.0,
            'biodiversity_index': len(self.species) / max(1, sum(s.population for s in self.species.values()) / 100),
            'trophic_diversity': len(set(s.species_type for s in self.species.values())),
            'avg_territory_size': np.mean([s.territory_size for s in self.species.values()]) if self.species else 0.0
        }
        
        self.evolution_history.append(state)
    
    def _record_speciation_event(self, species: Species):
        """Record a speciation event."""
        event = {
            'generation': self.generation,
            'species_id': species.species_id,
            'species_name': species.species_name,
            'species_type': species.species_type,
            'location': species.center,
            'initial_population': species.population
        }
        self.speciation_events.append(event)
    
    def _record_extinction_event(self, species: Species):
        """Record an extinction event."""
        event = {
            'generation': self.generation,
            'species_id': species.species_id,
            'species_name': species.species_name,
            'species_type': species.species_type,
            'final_population': species.population,
            'generations_alive': species.generation_alive,
            'final_fitness': species.fitness
        }
        self.extinction_events.append(event)
    
    def get_ecosystem_stats(self) -> Dict:
        """Get comprehensive ecosystem statistics."""
        species_by_type = defaultdict(int)
        for species in self.species.values():
            species_by_type[species.species_type] += 1
        
        return {
            'generation': self.generation,
            'num_species': len(self.species),
            'total_population': sum(s.population for s in self.species.values()),
            'species_by_type': dict(species_by_type),
            'num_relationships': len(self.relationships),
            'avg_fitness': np.mean([s.fitness for s in self.species.values()]) if self.species else 0.0,
            'num_environmental_zones': len(self.environmental_zones),
            'speciation_events': len(self.speciation_events),
            'extinction_events': len(self.extinction_events),
            'world_density': float(np.mean(self.ca_grid))
        }
    
    def generate_save_filename(self, prefix: str = "") -> str:
        """Generate descriptive filename for simulation saves."""
        timestamp = self.simulation_start_time.strftime("%Y%m%d_%H%M")
        
        stats = self.get_ecosystem_stats()
        num_species = stats['num_species']
        generation = stats['generation']
        total_pop = stats['total_population']
        
        # Most abundant species type
        species_types = stats['species_by_type']
        if species_types:
            dominant_type = max(species_types.items(), key=lambda x: x[1])[0]
        else:
            dominant_type = "none"
        
        filename = f"{prefix}ecosystem_sim_{timestamp}_gen{generation}_species{num_species}_pop{total_pop}_{dominant_type}_dominated"
        
        return filename

class EcosystemVisualizer:
    """Visualizer for the ecosystem evolution simulation."""
    
    def __init__(self, ecosystem: EcosystemSimulation):
        self.ecosystem = ecosystem
        
        plt.rcParams.update({
            'figure.facecolor': '#0a0a0a',
            'axes.facecolor': '#1a1a1a',
            'text.color': '#ffffff',
            'axes.labelcolor': '#ffffff',
        })
        
        self.fig = plt.figure(figsize=(22, 14))
        
        # Complex layout for ecosystem visualization
        gs = self.fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3)
        
        self.ax_ecosystem = self.fig.add_subplot(gs[0:3, 0:4])    # Main ecosystem view
        self.ax_food_web = self.fig.add_subplot(gs[0:2, 4:6])    # Food web network
        self.ax_population = self.fig.add_subplot(gs[2, 4])      # Population distribution
        self.ax_fitness = self.fig.add_subplot(gs[2, 5])        # Fitness distribution
        self.ax_timeline = self.fig.add_subplot(gs[3, :])       # Evolution timeline
        
        self.fig.patch.set_facecolor('#0a0a0a')
        for ax in [self.ax_ecosystem, self.ax_food_web, self.ax_population, self.ax_fitness, self.ax_timeline]:
            ax.set_facecolor('#1a1a1a')
        
        # Colors and styling
        self.cmap_ca = plt.matplotlib.colors.ListedColormap(['#0a0a0a', '#1a3a1a'])
        self.stats_history = []
        
    def update_visualization(self):
        """Update all visualization components."""
        # Clear all axes
        for ax in [self.ax_ecosystem, self.ax_food_web, self.ax_population, self.ax_fitness, self.ax_timeline]:
            ax.clear()
            ax.set_facecolor('#1a1a1a')
        
        self._draw_ecosystem_map()
        self._draw_food_web()
        self._draw_population_distribution()
        self._draw_fitness_distribution()
        self._draw_evolution_timeline()
    
    def _draw_ecosystem_map(self):
        """Draw the main ecosystem map."""
        # Draw CA background (habitat/terrain)
        self.ax_ecosystem.imshow(self.ecosystem.ca_grid, cmap=self.cmap_ca, 
                               interpolation='nearest', alpha=0.4)
        
        # Draw environmental zones
        for zone in self.ecosystem.environmental_zones.values():
            center_y, center_x = zone.center
            radius = zone.radius
            
            biome_info = BIOME_TYPES[zone.biome_type]
            circle = Circle((center_x, center_y), radius=radius, 
                          fill=False, color=biome_info['color'], 
                          linewidth=2, alpha=0.6, linestyle='--')
            self.ax_ecosystem.add_patch(circle)
            
            # Zone label
            self.ax_ecosystem.text(center_x, center_y + radius + 5, 
                                 f"{zone.biome_type.title()}\\n"
                                 f"T:{zone.temperature:.1f} "
                                 f"Bio:{zone.biodiversity_index:.2f}",
                                 ha='center', va='bottom', fontsize=7, 
                                 color=biome_info['color'], style='italic')
        
        # Draw species
        for species in self.ecosystem.species.values():
            species_info = SPECIES_TYPES[species.species_type]
            color = species_info['color']
            symbol = species_info['symbol']
            
            y, x = species.center
            
            # Territory circle (size based on territory and population)
            territory_radius = species.territory_size * (1 + np.log10(max(1, species.population)) / 3)
            territory_circle = Circle((x, y), radius=territory_radius, 
                                    fill=False, color=color, linewidth=1.5, 
                                    alpha=0.3 + species.fitness * 0.4)
            self.ax_ecosystem.add_patch(territory_circle)
            
            # Species marker
            marker_size = 100 + species.population // 10
            self.ax_ecosystem.scatter(x, y, c=color, s=marker_size, 
                                    alpha=0.8 + species.fitness * 0.2, 
                                    edgecolors='white', linewidth=1)
            
            # Species symbol
            self.ax_ecosystem.text(x, y, symbol, ha='center', va='center',
                                 fontsize=8, color='black', weight='bold')
            
            # Species info (for smaller populations, show name)
            if len(self.ecosystem.species) < 15:
                info_text = f"{species.species_name}\\n"
                info_text += f"Pop: {species.population}\\n"
                info_text += f"Fit: {species.fitness:.2f}"
                
                self.ax_ecosystem.text(x + territory_radius + 2, y, info_text,
                                     ha='left', va='center', fontsize=6, color=color,
                                     bbox=dict(boxstyle='round,pad=0.2', 
                                             facecolor='black', alpha=0.7))
        
        # Draw predation relationships as arrows
        for relationship in self.ecosystem.relationships.values():
            if relationship.relationship_type == 'predation':
                predator = self.ecosystem.species.get(relationship.predator_id)
                prey = self.ecosystem.species.get(relationship.prey_id)
                
                if predator and prey:
                    # Arrow from prey to predator (energy flow)
                    prey_x, prey_y = prey.center[1], prey.center[0]
                    pred_x, pred_y = predator.center[1], predator.center[0]
                    
                    # Calculate arrow position (offset from center)
                    dx, dy = pred_x - prey_x, pred_y - prey_y
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        dx, dy = dx/length, dy/length
                        
                        start_x = prey_x + dx * prey.territory_size
                        start_y = prey_y + dy * prey.territory_size
                        end_x = pred_x - dx * predator.territory_size
                        end_y = pred_y - dy * predator.territory_size
                        
                        alpha = relationship.strength * 0.6
                        width = relationship.strength * 1.5
                        
                        self.ax_ecosystem.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                                                 arrowprops=dict(arrowstyle='->', color='#ff4757',
                                                               alpha=alpha, lw=width))
        
        title = f'Ecosystem Evolution (Generation {self.ecosystem.generation})\\n'
        title += f'{len(self.ecosystem.species)} species, '
        title += f'{sum(s.population for s in self.ecosystem.species.values())} total population'
        
        self.ax_ecosystem.set_title(title, color='white', fontsize=12)
        self.ax_ecosystem.set_xlim(0, self.ecosystem.world_size[1])
        self.ax_ecosystem.set_ylim(0, self.ecosystem.world_size[0])
        self.ax_ecosystem.invert_yaxis()
    
    def _draw_food_web(self):
        """Draw the food web as a network graph."""
        if not self.ecosystem.species:
            self.ax_food_web.text(0.5, 0.5, 'No species yet', 
                                transform=self.ax_food_web.transAxes, ha='center', va='center',
                                color='white', fontsize=10)
            return
        
        # Create food web graph
        G = nx.DiGraph()
        
        # Add nodes
        for species in self.ecosystem.species.values():
            G.add_node(species.species_id, 
                      species_type=species.species_type,
                      population=species.population,
                      name=species.species_name)
        
        # Add edges (predation relationships)
        for relationship in self.ecosystem.relationships.values():
            if relationship.relationship_type == 'predation':
                G.add_edge(relationship.prey_id, relationship.predator_id,
                          strength=relationship.strength)
        
        if len(G.nodes()) == 0:
            return
        
        try:
            # Layout based on trophic levels
            pos = self._trophic_layout(G)
        except:
            try:
                pos = nx.spring_layout(G, k=1.5, iterations=50)
            except:
                pos = {node: (random.random(), random.random()) for node in G.nodes()}
        
        # Draw nodes
        for node_id in G.nodes():
            if node_id in self.ecosystem.species:
                species = self.ecosystem.species[node_id]
                color = SPECIES_TYPES[species.species_type]['color']
                size = 200 + species.population // 5
                
                self.ax_food_web.scatter(pos[node_id][0], pos[node_id][1], 
                                       c=color, s=size, alpha=0.8, 
                                       edgecolors='white', linewidth=1)
                
                # Node labels (species ID)
                self.ax_food_web.text(pos[node_id][0], pos[node_id][1], 
                                    str(node_id), ha='center', va='center', 
                                    fontsize=7, color='black', weight='bold')
        
        # Draw edges (predation arrows)
        for edge in G.edges(data=True):
            prey_id, predator_id, data = edge
            if prey_id in pos and predator_id in pos:
                strength = data.get('strength', 1.0)
                
                x_coords = [pos[prey_id][0], pos[predator_id][0]]
                y_coords = [pos[prey_id][1], pos[predator_id][1]]
                
                self.ax_food_web.annotate('', xy=(x_coords[1], y_coords[1]), 
                                        xytext=(x_coords[0], y_coords[0]),
                                        arrowprops=dict(arrowstyle='->', color='#ff6b6b',
                                                      alpha=strength*0.8, lw=strength*2))
        
        self.ax_food_web.set_title('Food Web Network', color='white', fontsize=10)
        self.ax_food_web.axis('off')
    
    def _trophic_layout(self, G):
        """Create layout based on trophic levels."""
        pos = {}
        
        # Group nodes by trophic level
        trophic_groups = defaultdict(list)
        for node_id in G.nodes():
            if node_id in self.ecosystem.species:
                species = self.ecosystem.species[node_id]
                trophic_level = SPECIES_TYPES[species.species_type]['trophic_level']
                trophic_groups[trophic_level].append(node_id)
        
        # Position nodes
        y_positions = {1: 0.2, 2: 0.4, 2.5: 0.5, 3: 0.7, 4: 0.9}
        
        for trophic_level, nodes in trophic_groups.items():
            y = y_positions.get(trophic_level, 0.6)
            
            if len(nodes) == 1:
                x = 0.5
            else:
                x_positions = np.linspace(0.1, 0.9, len(nodes))
                for i, node_id in enumerate(nodes):
                    pos[node_id] = (x_positions[i], y)
            
            if len(nodes) == 1:
                pos[nodes[0]] = (0.5, y)
        
        return pos
    
    def _draw_population_distribution(self):
        """Draw population distribution by species type."""
        stats = self.ecosystem.get_ecosystem_stats()
        species_by_type = stats['species_by_type']
        
        if sum(species_by_type.values()) > 0:
            types = list(species_by_type.keys())
            counts = list(species_by_type.values())
            colors = [SPECIES_TYPES[t]['color'] for t in types]
            
            bars = self.ax_population.bar(range(len(types)), counts, 
                                        color=colors, alpha=0.8)
            self.ax_population.set_xticks(range(len(types)))
            self.ax_population.set_xticklabels([t.replace('_', '\\n') for t in types], 
                                             rotation=0, ha='center', fontsize=7)
            self.ax_population.set_ylabel('Species Count', color='white', fontsize=8)
            self.ax_population.tick_params(colors='white', labelsize=7)
            
            # Add count labels
            for bar, count in zip(bars, counts):
                self.ax_population.text(bar.get_x() + bar.get_width()/2, 
                                      bar.get_height() + 0.1, str(count), 
                                      ha='center', va='bottom', fontsize=7, color='white')
        
        self.ax_population.set_title('Species by Type', color='white', fontsize=9)
    
    def _draw_fitness_distribution(self):
        """Draw fitness distribution of all species."""
        if not self.ecosystem.species:
            self.ax_fitness.text(0.5, 0.5, 'No species', 
                               transform=self.ax_fitness.transAxes, ha='center', va='center',
                               color='white', fontsize=9)
            return
        
        fitness_values = [species.fitness for species in self.ecosystem.species.values()]
        
        # Histogram of fitness values
        self.ax_fitness.hist(fitness_values, bins=10, color='#4ecdc4', alpha=0.7, edgecolor='white')
        self.ax_fitness.axvline(np.mean(fitness_values), color='#ff6b6b', 
                              linestyle='--', linewidth=2, label=f'Mean: {np.mean(fitness_values):.2f}')
        
        self.ax_fitness.set_xlabel('Fitness', color='white', fontsize=8)
        self.ax_fitness.set_ylabel('Count', color='white', fontsize=8)
        self.ax_fitness.set_title('Fitness Distribution', color='white', fontsize=9)
        self.ax_fitness.legend(fontsize=7, facecolor='#333333')
        self.ax_fitness.tick_params(colors='white', labelsize=7)
    
    def _draw_evolution_timeline(self):
        """Draw evolution timeline with multiple metrics."""
        stats = self.ecosystem.get_ecosystem_stats()
        self.stats_history.append(stats)
        
        if len(self.stats_history) > 1:
            generations = [s['generation'] for s in self.stats_history]
            
            # Multiple metrics
            num_species = [s['num_species'] for s in self.stats_history]
            total_population = [s['total_population'] / 100 for s in self.stats_history]  # Scale down
            avg_fitness = [s['avg_fitness'] for s in self.stats_history]
            num_relationships = [s['num_relationships'] for s in self.stats_history]
            
            # Plot metrics
            self.ax_timeline.plot(generations, num_species, label='Species Count', 
                                color='#ff6b6b', linewidth=2, marker='o', markersize=3)
            self.ax_timeline.plot(generations, total_population, label='Population (×100)',
                                color='#4ecdc4', linewidth=2, marker='s', markersize=3)
            self.ax_timeline.plot(generations, [f*20 for f in avg_fitness], label='Avg Fitness (×20)',
                                color='#feca57', linewidth=2, marker='^', markersize=3)
            self.ax_timeline.plot(generations, num_relationships, label='Relationships',
                                color='#ff9ff3', linewidth=2, marker='d', markersize=3)
        
        self.ax_timeline.set_xlabel('Generation', color='white')
        self.ax_timeline.set_ylabel('Count', color='white')
        self.ax_timeline.set_title('Ecosystem Evolution Timeline', color='white', fontsize=11)
        self.ax_timeline.legend(fontsize=8, facecolor='#333333', edgecolor='white')
        self.ax_timeline.grid(True, color='#333333', alpha=0.3)
        self.ax_timeline.tick_params(colors='white')
    
    def run_ecosystem_demo(self, max_generations: int = 300, steps_per_second: float = 3):
        """Run the ecosystem evolution demonstration."""
        generation_count = 0
        
        def animate(frame):
            nonlocal generation_count
            if generation_count < max_generations:
                # Ecosystem evolution
                self.ecosystem.step_ecosystem()
                
                # Update visualization
                self.update_visualization()
                
                # Progress logging
                if generation_count % 30 == 0:
                    stats = self.ecosystem.get_ecosystem_stats()
                    print(f"Gen {stats['generation']}: {stats['num_species']} species, "
                          f"{stats['total_population']} total pop, "
                          f"{stats['avg_fitness']:.2f} avg fitness")
                    
                    # Print sample species info
                    if stats['num_species'] > 0 and generation_count % 60 == 0:
                        print("\\nSample Species:")
                        for i, species in enumerate(list(self.ecosystem.species.values())[:3]):
                            print(f"  {species.species_name} ({species.species_type}): "
                                  f"Pop {species.population}, Fitness {species.fitness:.2f}")
                        
                        if self.ecosystem.relationships:
                            predation_count = sum(1 for r in self.ecosystem.relationships.values() 
                                                if r.relationship_type == 'predation')
                            print(f"\\nEcological relationships: {len(self.ecosystem.relationships)} "
                                  f"({predation_count} predation)")
                        print()
                
                generation_count += 1
            return []
        
        interval = int(1000 / steps_per_second)
        ani = animation.FuncAnimation(self.fig, animate, interval=interval, 
                                    blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        
        # Set up save functionality with descriptive filename
        def on_key(event):
            if event.key == 's':
                filename = self.ecosystem.generate_save_filename("ecosystem_snapshot_")
                self.fig.savefig(f"{filename}.png", dpi=150, bbox_inches='tight', 
                               facecolor='#0a0a0a', edgecolor='none')
                print(f"Saved screenshot as {filename}.png")
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.show()
        return ani
    
    def save_final_state(self):
        """Save the final state with descriptive filename."""
        filename = self.ecosystem.generate_save_filename("ecosystem_final_")
        self.fig.savefig(f"{filename}.png", dpi=150, bbox_inches='tight', 
                       facecolor='#0a0a0a', edgecolor='none')
        print(f"Final ecosystem state saved as {filename}.png")

def main():
    """Ecosystem evolution demonstration."""
    print("Ecosystem Evolution Simulation using CA-Graph Emergence")
    print("=" * 60)
    print("Features:")
    print("- CA patterns evolve into species with genetic traits")
    print("- Species form ecological relationships (predation, competition, mutualism)")
    print("- Environmental zones influence species evolution")
    print("- Coevolutionary pressure drives adaptation")
    print("- Population dynamics with carrying capacity")
    print("- Speciation and extinction events")
    print("- Food web visualization and trophic interactions")
    print("- Press 's' during animation to save snapshot")
    print()
    
    # Create ecosystem
    ecosystem = EcosystemSimulation(world_size=(120, 120))
    ecosystem.initialize_ecosystem(density=0.35, seed=42)
    
    # Create visualizer
    viz = EcosystemVisualizer(ecosystem)
    
    print("Environmental zones initialized:")
    for zone in ecosystem.environmental_zones.values():
        biome_info = BIOME_TYPES[zone.biome_type]
        print(f"  {zone.biome_type.title()}: productivity {biome_info['productivity']:.1f}, "
              f"temp {zone.temperature:.1f}")
    print()
    
    try:
        print("Starting ecosystem evolution...")
        print("Watch for:")
        print("- Species emerging from CA patterns")
        print("- Predator-prey relationships forming")
        print("- Population boom and bust cycles")
        print("- Evolutionary adaptations")
        print("- Speciation and extinction events")
        viz.run_ecosystem_demo(max_generations=250, steps_per_second=2.5)
        
        # Save final state
        viz.save_final_state()
        
    except KeyboardInterrupt:
        print("\\nEcosystem simulation stopped by user")
        viz.save_final_state()
    except Exception as e:
        print(f"Demo error: {e}")
        viz.save_final_state()

if __name__ == "__main__":
    main()