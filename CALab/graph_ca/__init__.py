"""
GraphCA - Cellular Automata on Graph Structures.

Enables emergent evolution of world state by operating CA rules
directly on knowledge graph nodes and edges.
"""

from CALab.graph_ca.core import (
    GraphCA,
    EvolutionConfig,
    EvolutionState,
    EvolutionPhase,
)

from CALab.graph_ca.rules import (
    species_evolution_rule,
    faction_evolution_rule,
    character_evolution_rule,
    location_evolution_rule,
    event_evolution_rule,
    get_default_rules,
    create_custom_rule,
)

from CALab.graph_ca.emergence import (
    EmergenceDetector,
    EmergencePattern,
    create_emergence_handler,
)

__all__ = [
    'GraphCA',
    'EvolutionConfig',
    'EvolutionState',
    'EvolutionPhase',
    'species_evolution_rule',
    'faction_evolution_rule',
    'character_evolution_rule',
    'location_evolution_rule',
    'event_evolution_rule',
    'get_default_rules',
    'create_custom_rule',
    'EmergenceDetector',
    'EmergencePattern',
    'create_emergence_handler',
]
