"""
Evolution Rules for GraphCA.

Defines update rules for different node types in the graph.
Each rule takes (node, neighbors, graph) and returns updated properties.
"""

from typing import Dict, List, Any, Optional, Callable
import random
import math


def species_evolution_rule(
    node: Any,
    neighbors: Dict[str, List[Any]],
    graph: 'KnowledgeGraph'
) -> Dict[str, Any]:
    """
    Evolution rule for species nodes.
    
    Updates:
    - population: Based on fitness, predation, and competition
    - fitness: Small random mutations
    - territory_size: Expands/contracts based on population
    """
    properties = node.properties
    updates = {}
    
    current_pop = properties.get('population', 100)
    fitness = properties.get('fitness', 0.5)
    territory = properties.get('territory_size', 10)
    
    predation_loss = 0
    competition_loss = 0
    mutualism_gain = 0
    
    for neighbor_data in neighbors.get('incoming', []):
        edge = neighbor_data['edge']
        neighbor = neighbor_data['node']
        
        if edge.edge_type == 'predation':
            predator_pop = neighbor.properties.get('population', 0)
            predator_fitness = neighbor.properties.get('fitness', 0.5)
            weight = edge.weight or 0.5
            
            predation_loss += weight * predator_pop * predator_fitness * 0.001
        
        elif edge.edge_type == 'competition':
            competitor_pop = neighbor.properties.get('population', 0)
            weight = edge.weight or 0.5
            
            competition_loss += weight * competitor_pop * 0.0005
    
    for neighbor_data in neighbors.get('outgoing', []):
        edge = neighbor_data['edge']
        neighbor = neighbor_data['node']
        
        if edge.edge_type == 'predation':
            prey_pop = neighbor.properties.get('population', 0)
            weight = edge.weight or 0.5
            
            food_gain = weight * prey_pop * 0.002
            mutualism_gain += food_gain
        
        elif edge.edge_type == 'mutualism':
            partner_pop = neighbor.properties.get('population', 0)
            weight = edge.weight or 0.5
            
            mutualism_gain += weight * min(partner_pop, current_pop) * 0.001
    
    growth_rate = fitness * 0.1
    natural_growth = current_pop * growth_rate * 0.1
    
    new_population = current_pop + natural_growth - predation_loss - competition_loss + mutualism_gain
    new_population = max(0, int(new_population))
    
    updates['population'] = new_population
    
    mutation = random.gauss(0, 0.02)
    selection_pressure = 1.0 - (new_population / (current_pop + 1))
    fitness_change = mutation + selection_pressure * 0.01
    
    new_fitness = max(0.1, min(1.0, fitness + fitness_change))
    updates['fitness'] = new_fitness
    
    if new_population > 0:
        ideal_territory = int(math.sqrt(new_population / 10))
        territory_change = (ideal_territory - territory) * 0.1
        new_territory = max(1, int(territory + territory_change))
        updates['territory_size'] = new_territory
    
    return updates


def faction_evolution_rule(
    node: Any,
    neighbors: Dict[str, List[Any]],
    graph: 'KnowledgeGraph'
) -> Dict[str, Any]:
    """
    Evolution rule for faction nodes.
    
    Updates:
    - power_level: Based on members, territories, conflicts
    - influence: Based on alliances and conflicts
    """
    properties = node.properties
    updates = {}
    
    current_power = properties.get('power_level', 0.5)
    
    member_count = 0
    territory_count = 0
    conflict_count = 0
    alliance_count = 0
    conflicts_won = 0
    conflicts_lost = 0
    
    for neighbor_data in neighbors.get('outgoing', []):
        edge = neighbor_data['edge']
        neighbor = neighbor_data['node']
        
        if edge.edge_type == 'contains':
            if neighbor.type == 'character':
                member_count += 1
            elif neighbor.type == 'location':
                territory_count += 1
        
        elif edge.edge_type == 'conflict':
            conflict_count += 1
            weight = edge.weight or 0.5
            if weight > 0.5:
                conflicts_won += 1
            else:
                conflicts_lost += 1
        
        elif edge.edge_type == 'alliance':
            alliance_count += 1
            partner_power = neighbor.properties.get('power_level', 0.5)
            current_power += partner_power * 0.05
    
    power_from_members = min(member_count * 0.05, 0.3)
    power_from_territory = min(territory_count * 0.03, 0.2)
    power_from_conflicts = conflicts_won * 0.05 - conflicts_lost * 0.08
    power_from_alliances = alliance_count * 0.02
    
    power_change = (
        power_from_members +
        power_from_territory +
        power_from_conflicts +
        power_from_alliances -
        0.02
    ) * 0.1
    
    new_power = current_power + power_change
    new_power = max(0.05, min(1.0, new_power))
    
    updates['power_level'] = new_power
    updates['member_count'] = member_count
    updates['territory_count'] = territory_count
    
    influence = (new_power * 0.5) + (alliance_count * 0.1) - (conflict_count * 0.05)
    updates['influence'] = max(0, min(1, influence))
    
    return updates


def character_evolution_rule(
    node: Any,
    neighbors: Dict[str, List[Any]],
    graph: 'KnowledgeGraph'
) -> Dict[str, Any]:
    """
    Evolution rule for character nodes.
    
    Updates:
    - stress: Based on conflicts, allies, recent events
    - influence: Based on faction membership and relationships
    """
    properties = node.properties
    updates = {}
    
    current_stress = properties.get('stress', 0.0)
    importance = properties.get('importance', 0.5)
    
    conflict_count = 0
    ally_count = 0
    faction_power = 0
    
    for neighbor_data in neighbors.get('incoming', []) + neighbors.get('outgoing', []):
        edge = neighbor_data['edge']
        neighbor = neighbor_data['node']
        
        if edge.edge_type == 'conflict':
            conflict_count += 1
        elif edge.edge_type == 'alliance':
            ally_count += 1
        elif edge.edge_type == 'contains' and neighbor.type == 'faction':
            faction_power = neighbor.properties.get('power_level', 0.5)
    
    stress_from_conflicts = conflict_count * 0.1
    stress_relief_from_allies = ally_count * 0.05
    stress_relief_from_faction = faction_power * 0.02
    
    stress_change = (
        stress_from_conflicts -
        stress_relief_from_allies -
        stress_relief_from_faction -
        0.01
    )
    
    new_stress = current_stress + stress_change * 0.1
    new_stress = max(0.0, min(1.0, new_stress))
    
    updates['stress'] = new_stress
    
    influence = importance * 0.5 + (1 - new_stress) * 0.3 + ally_count * 0.05
    updates['influence'] = max(0, min(1, influence))
    
    return updates


def location_evolution_rule(
    node: Any,
    neighbors: Dict[str, List[Any]],
    graph: 'KnowledgeGraph'
) -> Dict[str, Any]:
    """
    Evolution rule for location nodes.
    
    Updates:
    - population_density: Based on contained entities
    - danger_level: Based on conflicts in area
    - prosperity: Based on contained factions and resources
    """
    properties = node.properties
    updates = {}
    
    population = 0
    conflict_intensity = 0
    faction_count = 0
    
    for neighbor_data in neighbors.get('incoming', []):
        edge = neighbor_data['edge']
        neighbor = neighbor_data['node']
        
        if edge.edge_type == 'contains':
            if neighbor.type == 'character':
                population += 1
            elif neighbor.type == 'faction':
                faction_count += 1
            elif neighbor.type == 'species':
                population += neighbor.properties.get('population', 0) * 0.01
        
        elif edge.edge_type == 'conflict':
            conflict_intensity += edge.weight or 0.5
    
    updates['population_density'] = min(population / 100, 1.0)
    updates['danger_level'] = min(conflict_intensity / 5, 1.0)
    
    prosperity = (population / 100) * 0.3 + (1 - conflict_intensity / 10) * 0.4 + faction_count * 0.1
    updates['prosperity'] = max(0, min(1, prosperity))
    
    return updates


def event_evolution_rule(
    node: Any,
    neighbors: Dict[str, List[Any]],
    graph: 'KnowledgeGraph'
) -> Dict[str, Any]:
    """
    Evolution rule for event nodes.
    
    Updates:
    - significance: May decay over time
    - resolved: May be marked as resolved
    """
    properties = node.properties
    updates = {}
    
    significance = properties.get('significance', 0.5)
    event_type = properties.get('event_type', 'unknown')
    resolved = properties.get('resolved', False)
    
    if not resolved:
        decay_rate = 0.01
        new_significance = significance * (1 - decay_rate)
        
        if new_significance < 0.1:
            updates['resolved'] = True
            updates['resolution_time'] = 'natural_decay'
        
        updates['significance'] = new_significance
    
    return updates


DEFAULT_RULES: Dict[str, Callable] = {
    'species': species_evolution_rule,
    'faction': faction_evolution_rule,
    'character': character_evolution_rule,
    'location': location_evolution_rule,
    'event': event_evolution_rule,
}


def get_default_rules() -> Dict[str, Callable]:
    """Get all default evolution rules."""
    return DEFAULT_RULES.copy()


def create_custom_rule(
    update_functions: Dict[str, Callable]
) -> Callable:
    """
    Create a custom evolution rule from property update functions.
    
    Args:
        update_functions: Dict mapping property names to update functions
        
    Returns:
        Combined evolution rule
    """
    def custom_rule(node, neighbors, graph):
        updates = {}
        current_props = node.properties
        
        for prop_name, update_fn in update_functions.items():
            try:
                new_value = update_fn(current_props, neighbors, graph)
                updates[prop_name] = new_value
            except Exception:
                pass
        
        return updates
    
    return custom_rule
