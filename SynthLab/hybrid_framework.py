"""
Foundational Hybrid Architecture Framework for AlchemicalLab
============================================================

A modular system for composing CA, EA, NN, and Fractal components into
world-building simulations with emergent complexity.

Inspired by: Dwarf Fortress, Stellaris, Rimworld, Spore
Goal: Create substrate for emergent civilization/ecosystem games
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

# ============================================================================
# Core Component Interface
# ============================================================================

class HybridComponent(ABC):
    """Base class for all hybrid system components."""
    
    @abstractmethod
    def step(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one simulation step."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set component state."""
        pass


# ============================================================================
# Semantic Cellular Automaton Core
# ============================================================================

@dataclass
class CellState:
    """Rich cell state for world-building CA with multi-layer representation."""
    # Core properties
    alive: bool
    species: int
    energy: float

    # Biological layer
    age: int
    health: float
    reproduction_cooldown: int
    genetic_fitness: float

    # Economic layer
    wealth: float
    production_capacity: float
    trade_partners: jnp.ndarray  # Connection strengths to other cells
    market_demand: jnp.ndarray   # Demand for different goods

    # Cultural layer
    culture_vector: jnp.ndarray  # Cultural embedding
    language_complexity: float
    technological_level: float
    social_bonds: jnp.ndarray    # Relationships with neighbors

    # Physical layer
    terrain_type: int
    elevation: float
    moisture: float
    temperature: float
    resources: jnp.ndarray  # [minerals, water, food, energy]

    # Legacy properties (for compatibility)
    genetics: jnp.ndarray   # Genetic code
    color: jnp.ndarray      # RGB color for visualization
    language_tokens: jnp.ndarray  # Cultural/linguistic information
    connections: jnp.ndarray  # Connection weights to neighbors


class SemanticCA(HybridComponent):
    """GPU-accelerated cellular automaton with rich semantic properties."""
    
    def __init__(self,
                 grid_size: Tuple[int, int] = (100, 100),
                 n_species: int = 10,
                 n_cultures: int = 5,
                 n_terrain_types: int = 4,
                 n_resources: int = 4,  # minerals, water, food, energy
                 genetic_length: int = 32,
                 culture_dim: int = 8,
                 seed: int = 42):

        self.grid_size = grid_size
        self.n_species = n_species
        self.n_cultures = n_cultures
        self.n_terrain_types = n_terrain_types
        self.n_resources = n_resources
        self.genetic_length = genetic_length
        self.culture_dim = culture_dim

        # Initialize random key
        self.key = random.PRNGKey(seed)

        # Initialize grid state
        self._init_grid()

        # Default CA rules (will be evolved later)
        self.rules = self._default_rules()
    
    def _init_grid(self):
        """Initialize the multi-layer CA grid."""
        h, w = self.grid_size
        key = self.key

        # Split random keys for different properties
        keys = random.split(key, 15)

        # Initialize cell properties
        self.grid_state = {
            # Core properties
            'alive': random.bernoulli(keys[0], 0.3, (h, w)),
            'species': random.randint(keys[1], (h, w), 0, self.n_species),
            'energy': random.uniform(keys[2], (h, w), minval=0.0, maxval=1.0),

            # Biological layer
            'age': random.randint(keys[3], (h, w), 0, 100),
            'health': random.uniform(keys[4], (h, w), minval=0.5, maxval=1.0),
            'reproduction_cooldown': random.randint(keys[5], (h, w), 0, 20),
            'genetic_fitness': random.uniform(keys[6], (h, w), minval=0.0, maxval=1.0),

            # Economic layer
            'wealth': random.uniform(keys[7], (h, w), minval=0.0, maxval=0.5),
            'production_capacity': random.uniform(keys[8], (h, w), minval=0.1, maxval=1.0),
            'trade_partners': random.uniform(keys[9], (h, w, 8), minval=0.0, maxval=0.1),
            'market_demand': random.uniform(keys[10], (h, w, self.n_resources), minval=0.0, maxval=1.0),

            # Cultural layer
            'culture_vector': random.uniform(keys[11], (h, w, self.culture_dim), minval=-1.0, maxval=1.0),
            'language_complexity': random.uniform(keys[12], (h, w), minval=0.0, maxval=1.0),
            'technological_level': random.uniform(keys[13], (h, w), minval=0.0, maxval=0.5),
            'social_bonds': random.uniform(keys[14], (h, w, 8), minval=0.0, maxval=0.5),

            # Physical layer
            'terrain_type': random.randint(keys[0], (h, w), 0, self.n_terrain_types),  # Reuse key
            'elevation': random.uniform(keys[1], (h, w), minval=0.0, maxval=1.0),     # Reuse key
            'moisture': random.uniform(keys[2], (h, w), minval=0.0, maxval=1.0),      # Reuse key
            'temperature': random.uniform(keys[3], (h, w), minval=0.0, maxval=1.0),   # Reuse key
            'resources': random.uniform(keys[4], (h, w, self.n_resources), minval=0.0, maxval=1.0),  # Reuse key

            # Legacy properties (for compatibility)
            'genetics': random.randint(keys[5], (h, w, self.genetic_length), 0, 2),    # Reuse key
            'color': random.uniform(keys[6], (h, w, 3), minval=0.0, maxval=1.0),      # Reuse key
            'language_tokens': random.randint(keys[7], (h, w, 10), 0, 100),           # Reuse key
            'connections': random.uniform(keys[8], (h, w, 8), minval=-1.0, maxval=1.0)  # Reuse key
        }
    
    def _default_rules(self) -> Dict[str, Callable]:
        """Default CA transition rules for multi-layer simulation."""
        return {
            # Core rules
            'survival': self._survival_rule,
            'energy_flow': self._energy_flow_rule,

            # Biological rules
            'aging': self._aging_rule,
            'reproduction': self._reproduction_rule,
            'health': self._health_rule,
            'reproduction_cooldown': self._reproduction_cooldown_rule,
            'genetic_fitness': self._genetic_fitness_rule,

            # Economic rules
            'production': self._production_rule,
            'trade': self._trade_rule,
            'wealth_distribution': self._wealth_distribution_rule,
            'market_demand': self._market_demand_rule,

            # Cultural rules
            'culture_diffusion': self._culture_diffusion_rule,
            'language_evolution': self._language_evolution_rule,
            'technology_progress': self._technology_progress_rule,
            'social_bonds': self._social_bonds_rule,

            # Physical rules
            'resource_regeneration': self._resource_regeneration_rule,
            'environmental_effects': self._environmental_effects_rule,
            'moisture': self._moisture_rule,
            'temperature': self._temperature_rule,

            # Neural/Connection rules
            'connections': self._connections_rule,

            # Legacy rules (for compatibility)
            'resource_flow': self._resource_flow_rule,
            'genetic_drift': self._genetic_drift_rule,
            'language_spread': self._language_spread_rule,
            'color_evolution': self._color_evolution_rule
        }
    
    def _get_neighborhood(self, grid: jnp.ndarray, i: int, j: int) -> jnp.ndarray:
        """Get 3x3 neighborhood around cell (i,j) with wrapping."""
        h, w = grid.shape[:2]
        
        # Handle wrapping
        rows = jnp.array([(i-1) % h, i, (i+1) % h])
        cols = jnp.array([(j-1) % w, j, (j+1) % w])
        
        return grid[jnp.ix_(rows, cols)]
    
    def _survival_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Determine cell survival based on neighborhood."""
        # Count alive neighbors
        alive_neighbors = jnp.sum(neighborhood['alive']) - cell_state['alive']
        
        # Energy-based survival with Game of Life-like rules
        energy_factor = cell_state['energy']
        
        # Survival conditions
        survives = jnp.where(
            cell_state['alive'],
            # Alive cells: survive with 2-3 neighbors and sufficient energy
            (alive_neighbors >= 2) & (alive_neighbors <= 3) & (energy_factor > 0.2),
            # Dead cells: born with exactly 3 neighbors
            alive_neighbors == 3
        )
        
        return {'alive': survives}
    

    def _reproduction_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle reproduction and genetic inheritance."""
        if not cell_state['alive']:
            return cell_state
        
        # Find potential parents in neighborhood
        parent_mask = neighborhood['alive']
        
        # Simple genetic mixing (could be more sophisticated)
        new_genetics = jnp.where(
            random.bernoulli(self.key, 0.1, cell_state['genetics'].shape),  # 10% mutation rate
            1 - cell_state['genetics'],  # Flip bits for mutation
            cell_state['genetics']  # Keep original
        )
        
        return {'genetics': new_genetics}
    

    def _resource_flow_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle resource diffusion and consumption."""
        # Simple diffusion model
        neighbor_resources = jnp.mean(neighborhood['resources'], axis=(0, 1))
        diffusion_rate = 0.1
        
        new_resources = cell_state['resources'] + diffusion_rate * (
            neighbor_resources - cell_state['resources']
        )
        
        # Resource consumption if alive
        if cell_state['alive']:
            consumption = jnp.array([0.01, 0.005, 0.002, 0.001])  # minerals, water, food, energy
            new_resources = jnp.maximum(0.0, new_resources - consumption)
        
        return {'resources': new_resources}
    

    def _genetic_drift_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle genetic drift and species evolution."""
        if not cell_state['alive']:
            return cell_state
        
        # Count species in neighborhood
        neighbor_species = neighborhood['species'].flatten()
        # Find most common species (simplified)
        unique_species, counts = jnp.unique(neighbor_species, return_counts=True)
        dominant_species = unique_species[jnp.argmax(counts)]
        
        # Probability of species change based on genetic similarity
        # (simplified - could be much more sophisticated)
        species_pressure = 0.01
        
        new_species = jnp.where(
            random.uniform(self.key) < species_pressure,
            dominant_species,
            cell_state['species']
        )
        
        return {'species': new_species}
    

    def _language_spread_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle language/cultural information spread."""
        if not cell_state['alive']:
            return cell_state
        
        # Average language tokens from alive neighbors
        alive_mask = neighborhood['alive']
        if jnp.sum(alive_mask) > 0:
            # Simple averaging of alive neighbors
            alive_tokens = neighborhood['language_tokens'][alive_mask]
            avg_tokens = jnp.mean(alive_tokens, axis=0)

            # Mix current tokens with neighborhood average
            mixing_rate = 0.05
            new_tokens = (1 - mixing_rate) * cell_state['language_tokens'] + \
                        mixing_rate * avg_tokens
        else:
            new_tokens = cell_state['language_tokens']
        
        return {'language_tokens': new_tokens}
    

    def _color_evolution_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Evolve cell colors based on genetics and environment."""
        if not cell_state['alive']:
            return {'color': jnp.array([0.0, 0.0, 0.0])}  # Dead cells are black

        # Map genetics to color (simplified)
        # Reshape to groups of 3, padding if necessary
        genetics_flat = cell_state['genetics'].flatten()
        n_groups = len(genetics_flat) // 3
        if n_groups > 0:
            genetic_color = jnp.mean(genetics_flat[:n_groups*3].reshape(n_groups, 3), axis=0)
        else:
            genetic_color = jnp.array([0.5, 0.5, 0.5])  # default

        # Environmental pressure on color
        neighbor_colors = jnp.mean(neighborhood['color'], axis=(0, 1))
        adaptation_rate = 0.02

        new_color = (1 - adaptation_rate) * genetic_color + \
                   adaptation_rate * neighbor_colors

        return {'color': jnp.clip(new_color, 0.0, 1.0)}

    # Multi-layer rule implementations


    def _energy_flow_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle energy flow and consumption."""
        if not cell_state['alive']:
            return {'energy': 0.0}

        # Energy consumption based on activity
        base_consumption = 0.01
        age_penalty = cell_state['age'] * 0.001
        health_bonus = (1.0 - cell_state['health']) * 0.005

        consumption = base_consumption + age_penalty - health_bonus

        # Energy from resources
        resource_energy = jnp.sum(cell_state['resources']) * 0.1

        # Energy from neighbors (diffusion)
        neighbor_energy = jnp.mean(neighborhood['energy'][neighborhood['alive']])
        if jnp.sum(neighborhood['alive']) == 0:
            neighbor_energy = 0.0

        diffusion_rate = 0.05
        new_energy = cell_state['energy'] + resource_energy + diffusion_rate * (neighbor_energy - cell_state['energy'])
        new_energy = jnp.maximum(0.0, new_energy - consumption)

        return {'energy': jnp.clip(new_energy, 0.0, 1.0)}


    def _aging_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle aging process."""
        if not cell_state['alive']:
            return {'age': cell_state['age']}

        # Age increases by 1 each step
        new_age = cell_state['age'] + 1

        # Death from old age (simplified)
        max_age = 200 + cell_state['genetic_fitness'] * 100  # Genetic fitness extends lifespan
        if new_age > max_age:
            return {'age': new_age, 'alive': False}

        return {'age': new_age}


    def _health_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle health dynamics."""
        if not cell_state['alive']:
            return {'health': 0.0}

        # Health affected by energy, resources, and environment
        energy_factor = cell_state['energy']
        resource_factor = jnp.mean(cell_state['resources'])
        environmental_factor = 1.0 - jnp.abs(cell_state['temperature'] - 0.5)  # Optimal temp around 0.5

        # Disease spread from unhealthy neighbors
        unhealthy_neighbors = jnp.sum(neighborhood['health'] < 0.3)
        disease_risk = unhealthy_neighbors * 0.02

        new_health = cell_state['health'] + 0.05 * (energy_factor + resource_factor + environmental_factor) - disease_risk
        new_health = jnp.clip(new_health, 0.0, 1.0)

        return {'health': new_health}


    def _production_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle production of goods."""
        if not cell_state['alive']:
            return {'production_capacity': cell_state['production_capacity']}

        # Production based on health, technology, and terrain
        base_production = cell_state['production_capacity'] * cell_state['health']
        tech_bonus = cell_state['technological_level'] * 0.5

        # Terrain modifiers
        terrain_modifiers = jnp.array([0.5, 1.0, 1.5, 0.8])  # Different terrain types
        terrain_bonus = terrain_modifiers[cell_state['terrain_type']]

        new_production = base_production * (1.0 + tech_bonus) * terrain_bonus

        return {'production_capacity': jnp.clip(new_production, 0.0, 2.0)}


    def _trade_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle trade between neighboring cells."""
        if not cell_state['alive']:
            return {'trade_partners': cell_state['trade_partners']}

        # Simple trade: increase trade partners with wealthy neighbors
        neighbor_wealth = neighborhood['wealth']
        neighbor_alive = neighborhood['alive']

        # Trade potential based on neighbor wealth
        trade_potential = jnp.where(neighbor_alive, neighbor_wealth, 0.0)

        # Update trade partner strengths (simplified)
        learning_rate = 0.05
        # Aggregate trade potential across all neighbors
        avg_trade_potential = jnp.mean(trade_potential)
        trade_update = jnp.full_like(cell_state['trade_partners'], avg_trade_potential)
        new_trade_partners = cell_state['trade_partners'] + learning_rate * (trade_update - cell_state['trade_partners'])

        return {'trade_partners': jnp.clip(new_trade_partners, 0.0, 1.0)}


    def _wealth_distribution_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle wealth distribution and economic interactions."""
        if not cell_state['alive']:
            return {'wealth': 0.0}

        # Wealth from production and trade
        production_income = cell_state['production_capacity'] * 0.1
        trade_income = jnp.sum(cell_state['trade_partners'] * 0.05)

        # Taxation/redistribution
        total_neighbor_wealth = jnp.sum(neighborhood['wealth'] * neighborhood['alive'])
        n_neighbors = jnp.sum(neighborhood['alive'])
        avg_neighbor_wealth = total_neighbor_wealth / jnp.maximum(n_neighbors, 1)

        # Progressive taxation: richer cells give to poorer ones
        wealth_gap = cell_state['wealth'] - avg_neighbor_wealth
        redistribution = jnp.where(wealth_gap > 0.1, wealth_gap * 0.02, 0.0)

        new_wealth = cell_state['wealth'] + production_income + trade_income - redistribution

        return {'wealth': jnp.maximum(0.0, new_wealth)}


    def _culture_diffusion_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle cultural diffusion between cells."""
        if not cell_state['alive']:
            return {'culture_vector': cell_state['culture_vector']}

        # Cultural exchange with alive neighbors
        alive_mask = neighborhood['alive']
        if jnp.sum(alive_mask) == 0:
            return {'culture_vector': cell_state['culture_vector']}

        neighbor_cultures = neighborhood['culture_vector']
        avg_neighbor_culture = jnp.mean(neighbor_cultures, axis=(0, 1))

        # Cultural assimilation rate
        assimilation_rate = 0.02 * cell_state['language_complexity']

        new_culture = cell_state['culture_vector'] + assimilation_rate * (avg_neighbor_culture - cell_state['culture_vector'])

        # Normalize culture vector
        new_culture = new_culture / jnp.maximum(jnp.linalg.norm(new_culture), 1e-8)

        return {'culture_vector': new_culture}


    def _language_evolution_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle language evolution and complexity."""
        if not cell_state['alive']:
            return {'language_complexity': 0.0}

        # Language complexity increases with social interactions
        social_interactions = jnp.sum(cell_state['social_bonds'])
        learning_rate = 0.001

        new_complexity = cell_state['language_complexity'] + learning_rate * social_interactions

        # Complexity also affected by technological level
        tech_bonus = cell_state['technological_level'] * 0.1

        new_complexity = jnp.clip(new_complexity + tech_bonus, 0.0, 1.0)

        return {'language_complexity': new_complexity}


    def _technology_progress_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle technological progress."""
        if not cell_state['alive']:
            return {'technological_level': cell_state['technological_level']}

        # Technology advances through research and cultural exchange
        research_rate = 0.002 * cell_state['wealth'] * cell_state['language_complexity']

        # Knowledge diffusion from neighbors
        neighbor_tech = neighborhood['technological_level']
        avg_neighbor_tech = jnp.mean(neighbor_tech)

        diffusion_rate = 0.01
        tech_transfer = diffusion_rate * (avg_neighbor_tech - cell_state['technological_level'])

        new_tech = cell_state['technological_level'] + research_rate + tech_transfer

        return {'technological_level': jnp.clip(new_tech, 0.0, 1.0)}


    def _resource_regeneration_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle resource regeneration based on environment."""
        # Resources regenerate based on terrain and environmental conditions
        terrain_regeneration = jnp.array([0.01, 0.02, 0.005, 0.015])  # Different rates per terrain
        base_regen = terrain_regeneration[cell_state['terrain_type']]

        # Environmental modifiers
        moisture_bonus = cell_state['moisture'] * 0.01
        temperature_factor = 1.0 - jnp.abs(cell_state['temperature'] - 0.6)  # Optimal around 0.6

        regen_rate = base_regen * (1.0 + moisture_bonus) * temperature_factor

        # Apply regeneration
        new_resources = cell_state['resources'] + regen_rate

        # Cap at maximum
        new_resources = jnp.clip(new_resources, 0.0, 1.0)

        return {'resources': new_resources}


    def _environmental_effects_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle environmental effects on cells."""
        if not cell_state['alive']:
            return {'health': cell_state['health']}

        # Environmental stress affects health
        temp_stress = jnp.abs(cell_state['temperature'] - 0.5) * 0.1
        moisture_stress = jnp.abs(cell_state['moisture'] - 0.5) * 0.05

        environmental_stress = temp_stress + moisture_stress

        # Health reduction from environmental stress
        health_penalty = environmental_stress * 0.02
        new_health = jnp.maximum(0.0, cell_state['health'] - health_penalty)

        return {'health': new_health}

    def _reproduction_cooldown_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle reproduction cooldown timing."""
        if not cell_state['alive']:
            return {'reproduction_cooldown': cell_state['reproduction_cooldown']}

        # Decrease cooldown by 1 each step, minimum 0
        new_cooldown = jnp.maximum(0, cell_state['reproduction_cooldown'] - 1)

        return {'reproduction_cooldown': new_cooldown}

    def _genetic_fitness_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Update genetic fitness based on survival and performance."""
        if not cell_state['alive']:
            return {'genetic_fitness': cell_state['genetic_fitness']}

        # Fitness based on health, energy, age survival, and resource abundance
        health_contribution = cell_state['health'] * 0.3
        energy_contribution = cell_state['energy'] * 0.3
        resource_contribution = jnp.mean(cell_state['resources']) * 0.2
        age_bonus = jnp.where(cell_state['age'] > 50, 0.1, 0.0)  # Survival bonus
        tech_bonus = cell_state['technological_level'] * 0.1

        new_fitness = health_contribution + energy_contribution + resource_contribution + age_bonus + tech_bonus
        new_fitness = jnp.clip(new_fitness, 0.0, 1.0)

        # Gradual adaptation
        adaptation_rate = 0.05
        updated_fitness = cell_state['genetic_fitness'] + adaptation_rate * (new_fitness - cell_state['genetic_fitness'])

        return {'genetic_fitness': updated_fitness}

    def _market_demand_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Update market demand based on resource needs."""
        if not cell_state['alive']:
            return {'market_demand': cell_state['market_demand']}

        # Demand increases when resources are low
        resource_shortage = 1.0 - cell_state['resources']  # Higher when resources are low
        base_demand = 0.5

        # Update demand with some inertia
        demand_rate = 0.1
        new_demand = cell_state['market_demand'] + demand_rate * (base_demand + resource_shortage * 0.5 - cell_state['market_demand'])
        new_demand = jnp.clip(new_demand, 0.0, 1.0)

        return {'market_demand': new_demand}

    def _social_bonds_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Update social bonds with neighboring cells."""
        if not cell_state['alive']:
            return {'social_bonds': cell_state['social_bonds']}

        # Extract 8 neighbors (Moore neighborhood excluding center)
        indices = jnp.array([0,1,2,3,5,6,7,8])
        neighbor_species = neighborhood['species'].flatten()[indices]
        neighbor_alive = neighborhood['alive'].flatten()[indices]
        neighbor_wealth = neighborhood['wealth'].flatten()[indices]

        # Species similarity (1.0 if same species, 0.0 if different)
        species_similarity = jnp.where(neighbor_species == cell_state['species'], 1.0, 0.0)
        species_similarity = jnp.where(neighbor_alive, species_similarity, 0.0)

        # Wealth similarity (closer wealth = stronger bonds)
        wealth_diff = jnp.abs(neighbor_wealth - cell_state['wealth'])
        wealth_similarity = jnp.maximum(0.0, 1.0 - wealth_diff)

        # Combined social attraction
        social_attraction = (species_similarity * 0.6 + wealth_similarity * 0.4) * neighbor_alive

        # Update bonds with learning rate
        learning_rate = 0.05
        new_bonds = cell_state['social_bonds'] + learning_rate * (social_attraction - cell_state['social_bonds'])
        new_bonds = jnp.clip(new_bonds, 0.0, 1.0)

        return {'social_bonds': new_bonds}

    def _moisture_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle moisture dynamics and diffusion."""
        # Moisture diffuses between neighboring cells
        neighbor_moisture = neighborhood['moisture']
        neighbor_alive = neighborhood['alive']

        # Average moisture from alive neighbors
        alive_moisture = jnp.where(neighbor_alive, neighbor_moisture, cell_state['moisture'])
        avg_neighbor_moisture = jnp.mean(alive_moisture)

        # Diffusion rate
        diffusion_rate = 0.02
        new_moisture = cell_state['moisture'] + diffusion_rate * (avg_neighbor_moisture - cell_state['moisture'])

        # Terrain affects moisture retention
        terrain_retention = jnp.array([0.8, 1.0, 0.9, 0.7])  # Different retention per terrain
        retention_factor = terrain_retention[cell_state['terrain_type']]
        new_moisture = new_moisture * retention_factor

        # Environmental evaporation/precipitation (simplified)
        elevation_factor = cell_state['elevation'] * 0.1  # Higher elevation = less moisture
        new_moisture = new_moisture - elevation_factor

        new_moisture = jnp.clip(new_moisture, 0.0, 1.0)

        return {'moisture': new_moisture}

    def _temperature_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Handle temperature dynamics and heat diffusion."""
        # Temperature diffuses between cells
        neighbor_temp = neighborhood['temperature']
        neighbor_alive = neighborhood['alive']

        # Average temperature from neighbors
        alive_temp = jnp.where(neighbor_alive, neighbor_temp, cell_state['temperature'])
        avg_neighbor_temp = jnp.mean(alive_temp)

        # Heat diffusion
        diffusion_rate = 0.03
        new_temp = cell_state['temperature'] + diffusion_rate * (avg_neighbor_temp - cell_state['temperature'])

        # Elevation affects temperature (higher = cooler)
        elevation_cooling = cell_state['elevation'] * 0.2
        new_temp = new_temp - elevation_cooling

        # Moisture affects temperature (wetter = cooler)
        moisture_cooling = cell_state['moisture'] * 0.1
        new_temp = new_temp - moisture_cooling

        new_temp = jnp.clip(new_temp, 0.0, 1.0)

        return {'temperature': new_temp}

    def _connections_rule(self, cell_state: Dict, neighborhood: Dict) -> Dict:
        """Update connection weights between cells."""
        if not cell_state['alive']:
            return {'connections': cell_state['connections']}

        # Extract 8 neighbors
        indices = jnp.array([0,1,2,3,5,6,7,8])
        neighbor_alive = neighborhood['alive'].flatten()[indices]
        neighbor_species = neighborhood['species'].flatten()[indices]
        # For multi-dimensional properties, reshape spatial dims and index
        culture_dim = neighborhood['culture_vector'].shape[-1]
        neighbor_culture = neighborhood['culture_vector'].reshape(9, culture_dim)[indices]  # Shape: (8, culture_dim)

        # Species-based connection strength
        species_match = jnp.where(neighbor_species == cell_state['species'], 1.0, 0.0)

        # Cultural similarity
        culture_diff = jnp.linalg.norm(neighbor_culture - cell_state['culture_vector'], axis=-1)
        culture_similarity = jnp.maximum(0.0, 1.0 - culture_diff / 2.0)  # Normalize diff

        # Combined connection strength
        connection_strength = (species_match * 0.5 + culture_similarity * 0.5) * neighbor_alive

        # Update connections with hebbian learning
        learning_rate = 0.02
        new_connections = cell_state['connections'] + learning_rate * (connection_strength - cell_state['connections'])
        new_connections = jnp.clip(new_connections, -1.0, 1.0)

        return {'connections': new_connections}
    
    def step(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one CA simulation step."""
        h, w = self.grid_size

        # Create new state
        new_state = {}

        # Initialize new state arrays
        for prop in self.grid_state.keys():
            new_state[prop] = jnp.zeros_like(self.grid_state[prop])

        # Apply rules to each cell
        for i in range(h):
            for j in range(w):
                # Get cell state and neighborhood
                cell = {k: v[i, j] for k, v in self.grid_state.items()}

                # Get neighborhood for all properties
                neighborhood = {}
                for prop, values in self.grid_state.items():
                    neighborhood[prop] = self._get_neighborhood(values, i, j)

                # Apply all rules
                updates = {}
                for rule_name, rule_func in self.rules.items():
                    rule_updates = rule_func(cell, neighborhood)
                    updates.update(rule_updates)

                # Update cell state
                for prop, value in updates.items():
                    if prop in new_state:
                        new_state[prop] = new_state[prop].at[i, j].set(value)

        self.grid_state = new_state
        return self.grid_state
    
    def get_state(self) -> Dict[str, Any]:
        """Get current CA state."""
        return self.grid_state.copy()
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set CA state."""
        self.grid_state = state.copy()
    
    def get_visualization_data(self) -> Dict[str, jnp.ndarray]:
        """Get data for visualization."""
        return {
            'alive_grid': self.grid_state['alive'].astype(jnp.float32),
            'species_grid': self.grid_state['species'].astype(jnp.float32),
            'energy_grid': self.grid_state['energy'],
            'color_grid': self.grid_state['color'],
            'resource_grid': jnp.sum(self.grid_state['resources'], axis=-1)
        }


# ============================================================================
# Hybrid System Orchestrator
# ============================================================================

class HybridSystem:
    """Main system orchestrator for hybrid components."""
    
    def __init__(self):
        self.components = {}
        self.connections = {}
        self.global_state = {}
    
    def add_component(self, name: str, component: HybridComponent):
        """Add a component to the system."""
        self.components[name] = component
    
    def connect(self, source: str, target: str, transform: Optional[Callable] = None):
        """Connect output of source component to input of target component."""
        if source not in self.connections:
            self.connections[source] = []
        
        self.connections[source].append({
            'target': target,
            'transform': transform or (lambda x: x)
        })
    
    def step(self):
        """Execute one simulation step for all components."""
        # Collect all component states
        context = {name: comp.get_state() for name, comp in self.components.items()}
        context.update(self.global_state)
        
        # Update each component
        for name, component in self.components.items():
            new_state = component.step(component.get_state(), context)
            
            # Propagate outputs to connected components
            if name in self.connections:
                for connection in self.connections[name]:
                    target = connection['target']
                    transform = connection['transform']
                    
                    if target in self.components:
                        # Apply transformation and update target
                        transformed_data = transform(new_state)
                        # Here you would implement the actual data flow logic
    
    def run(self, steps: int):
        """Run simulation for specified number of steps."""
        for i in range(steps):
            self.step()
            if i % 100 == 0:  # Progress indicator
                print(f"Step {i}/{steps}")


# ============================================================================
# Example Usage & Testing
# ============================================================================

if __name__ == "__main__":
    # Create a simple world-building simulation
    print("Initializing AlchemicalLab Hybrid System...")
    
    # Create semantic CA
    world_ca = SemanticCA(grid_size=(10, 10), seed=42)
    
    # Create hybrid system
    system = HybridSystem()
    system.add_component("world", world_ca)
    
    # Run simulation
    print("Running world simulation...")
    for step in range(10):
        world_state = world_ca.step({}, {})
        
        # Simple statistics
        alive_count = jnp.sum(world_state['alive'])
        avg_energy = jnp.mean(world_state['energy'])
        species_diversity = len(jnp.unique(world_state['species']))
        
        print(f"Step {step}: Alive={alive_count}, Energy={avg_energy:.3f}, Species={species_diversity}")
    
    print("Simulation complete!")
    
    # Get visualization data
    viz_data = world_ca.get_visualization_data()
    print(f"Visualization data keys: {list(viz_data.keys())}")
    print(f"Grid shape: {viz_data['alive_grid'].shape}")