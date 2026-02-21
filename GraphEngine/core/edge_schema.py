"""
Edge Schema Definitions for GraphEngine.

Defines edge types, validation rules, and relationship constraints.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class EdgeTypeDefinition:
    """Definition of an edge type."""
    name: str
    source_types: List[str]  # Valid source node types, ["*"] for any
    target_types: List[str]  # Valid target node types, ["*"] for any
    weight_range: Optional[Tuple[float, float]] = None  # (min, max) or None for unweighted
    bidirectional: bool = False
    description: str = ""
    inverse_type: Optional[str] = None  # Name of inverse edge type
    
    def validate_weight(self, weight: Optional[float]) -> Tuple[bool, Optional[float]]:
        """Validate and normalize weight value."""
        if self.weight_range is None:
            if weight is not None:
                return False, None  # Should not have weight
            return True, None
        
        if weight is None:
            return True, (self.weight_range[0] + self.weight_range[1]) / 2
        
        min_w, max_w = self.weight_range
        if not (min_w <= weight <= max_w):
            return False, max(min_w, min(max_w, weight))
        
        return True, weight


# Built-in edge type definitions
EDGE_TYPE_DEFINITIONS: Dict[str, EdgeTypeDefinition] = {
    # Ecological relationships
    'predation': EdgeTypeDefinition(
        name='predation',
        source_types=['species'],
        target_types=['species'],
        weight_range=(0.0, 1.0),
        bidirectional=False,
        description='Predator-prey relationship',
        inverse_type='prey_of'
    ),
    
    'prey_of': EdgeTypeDefinition(
        name='prey_of',
        source_types=['species'],
        target_types=['species'],
        weight_range=(0.0, 1.0),
        bidirectional=False,
        description='Inverse of predation',
        inverse_type='predation'
    ),
    
    'competition': EdgeTypeDefinition(
        name='competition',
        source_types=['species', 'faction'],
        target_types=['species', 'faction'],
        weight_range=(0.0, 1.0),
        bidirectional=True,
        description='Resource competition'
    ),
    
    'mutualism': EdgeTypeDefinition(
        name='mutualism',
        source_types=['species'],
        target_types=['species'],
        weight_range=(0.0, 1.0),
        bidirectional=True,
        description='Beneficial symbiotic relationship'
    ),
    
    # Narrative relationships
    'alliance': EdgeTypeDefinition(
        name='alliance',
        source_types=['character', 'faction'],
        target_types=['character', 'faction'],
        weight_range=(0.0, 1.0),
        bidirectional=True,
        description='Cooperative relationship'
    ),
    
    'conflict': EdgeTypeDefinition(
        name='conflict',
        source_types=['character', 'faction', 'species'],
        target_types=['character', 'faction', 'species'],
        weight_range=(0.0, 1.0),
        bidirectional=False,
        description='Antagonistic relationship',
        inverse_type='threatened_by'
    ),
    
    'threatened_by': EdgeTypeDefinition(
        name='threatened_by',
        source_types=['character', 'faction', 'species'],
        target_types=['character', 'faction', 'species'],
        weight_range=(0.0, 1.0),
        bidirectional=False,
        description='Inverse of conflict',
        inverse_type='conflict'
    ),
    
    'dependency': EdgeTypeDefinition(
        name='dependency',
        source_types=['character', 'faction', 'species'],
        target_types=['location', 'species', 'concept'],
        weight_range=(0.0, 1.0),
        bidirectional=False,
        description='Reliance relationship'
    ),
    
    # Spatial/Structural relationships
    'contains': EdgeTypeDefinition(
        name='contains',
        source_types=['location', 'faction'],
        target_types=['character', 'species', 'location', 'event'],
        weight_range=None,
        bidirectional=False,
        description='Containment relationship',
        inverse_type='contained_by'
    ),
    
    'contained_by': EdgeTypeDefinition(
        name='contained_by',
        source_types=['character', 'species', 'location', 'event'],
        target_types=['location', 'faction'],
        weight_range=None,
        bidirectional=False,
        description='Inverse of contains',
        inverse_type='contains'
    ),
    
    'adjacent': EdgeTypeDefinition(
        name='adjacent',
        source_types=['location'],
        target_types=['location'],
        weight_range=None,
        bidirectional=True,
        description='Physical proximity'
    ),
    
    'origin': EdgeTypeDefinition(
        name='origin',
        source_types=['character', 'species'],
        target_types=['location'],
        weight_range=(0.0, 1.0),
        bidirectional=False,
        description='Origin location'
    ),
    
    # Semantic relationships
    'references': EdgeTypeDefinition(
        name='references',
        source_types=['event', 'concept', 'character'],
        target_types=['character', 'location', 'event', 'concept'],
        weight_range=(0.0, 1.0),
        bidirectional=False,
        description='Mentions or references'
    ),
    
    'similar_to': EdgeTypeDefinition(
        name='similar_to',
        source_types=['*'],
        target_types=['*'],
        weight_range=(0.0, 1.0),
        bidirectional=True,
        description='Semantic similarity'
    ),
    
    'causes': EdgeTypeDefinition(
        name='causes',
        source_types=['event', 'pattern'],
        target_types=['event', 'character', 'species'],
        weight_range=(0.0, 1.0),
        bidirectional=False,
        description='Causal relationship',
        inverse_type='caused_by'
    ),
    
    'caused_by': EdgeTypeDefinition(
        name='caused_by',
        source_types=['event', 'character', 'species'],
        target_types=['event', 'pattern'],
        weight_range=(0.0, 1.0),
        bidirectional=False,
        description='Inverse of causes',
        inverse_type='causes'
    ),
    
    # Special relationships for CA integration
    'evolves_to': EdgeTypeDefinition(
        name='evolves_to',
        source_types=['species', 'pattern'],
        target_types=['species', 'pattern'],
        weight_range=(0.0, 1.0),
        bidirectional=False,
        description='Evolutionary transition'
    ),
    
    'influences': EdgeTypeDefinition(
        name='influences',
        source_types=['pattern', 'event', 'concept'],
        target_types=['character', 'species', 'location', 'faction'],
        weight_range=(0.0, 1.0),
        bidirectional=False,
        description='Influence relationship'
    ),
}


class EdgeSchema:
    """Schema validation and management for edges."""
    
    def __init__(self, custom_definitions: Optional[Dict[str, EdgeTypeDefinition]] = None):
        self.definitions = EDGE_TYPE_DEFINITIONS.copy()
        if custom_definitions:
            self.definitions.update(custom_definitions)
    
    def get_definition(self, edge_type: str) -> Optional[EdgeTypeDefinition]:
        """Get the definition for an edge type."""
        return self.definitions.get(edge_type)
    
    def list_types(self) -> List[str]:
        """List all known edge types."""
        return list(self.definitions.keys())
    
    def validate(
        self,
        edge_type: str,
        source_type: str,
        target_type: str,
        weight: Optional[float] = None
    ) -> tuple:
        """
        Validate an edge against schema.
        
        Returns:
            (is_valid, errors, normalized_weight, bidirectional)
        """
        definition = self.definitions.get(edge_type)
        
        if not definition:
            return False, [f"Unknown edge type: {edge_type}"], weight, False
        
        errors = []
        
        # Validate source type
        if definition.source_types != ["*"]:
            if source_type not in definition.source_types:
                errors.append(
                    f"Edge '{edge_type}' cannot have '{source_type}' as source. "
                    f"Valid types: {definition.source_types}"
                )
        
        # Validate target type
        if definition.target_types != ["*"]:
            if target_type not in definition.target_types:
                errors.append(
                    f"Edge '{edge_type}' cannot have '{target_type}' as target. "
                    f"Valid types: {definition.target_types}"
                )
        
        # Validate weight
        weight_valid, normalized_weight = definition.validate_weight(weight)
        if not weight_valid:
            errors.append(f"Invalid weight {weight} for edge type '{edge_type}'")
        
        return len(errors) == 0, errors, normalized_weight, definition.bidirectional
    
    def get_compatible_edges(
        self,
        source_type: str,
        target_type: str
    ) -> List[str]:
        """Get list of edge types compatible with given node types."""
        compatible = []
        
        for name, definition in self.definitions.items():
            source_ok = (
                definition.source_types == ["*"] or
                source_type in definition.source_types
            )
            target_ok = (
                definition.target_types == ["*"] or
                target_type in definition.target_types
            )
            
            if source_ok and target_ok:
                compatible.append(name)
        
        return compatible
    
    def get_inverse(self, edge_type: str) -> Optional[str]:
        """Get the inverse edge type name."""
        definition = self.definitions.get(edge_type)
        return definition.inverse_type if definition else None
    
    def register_type(self, definition: EdgeTypeDefinition):
        """Register a new edge type definition."""
        self.definitions[definition.name] = definition


DEFAULT_SCHEMA = EdgeSchema()


def validate_edge(
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: Optional[float] = None
) -> tuple:
    """
    Validate an edge.
    
    Args:
        source_id: Source node ID (or type hint like 'character:')
        target_id: Target node ID
        edge_type: Type of edge
        weight: Optional weight
    
    Returns:
        (is_valid, errors) tuple
    """
    source_type = source_id.split(':')[0] if ':' in source_id else '*'
    target_type = target_id.split(':')[0] if ':' in target_id else '*'
    
    is_valid, errors, _, _ = DEFAULT_SCHEMA.validate(
        edge_type, source_type, target_type, weight
    )
    return is_valid, errors


def get_edge_semantics(edge_type: str) -> Dict[str, any]:
    """Get semantic information about an edge type."""
    definition = EDGE_TYPE_DEFINITIONS.get(edge_type)
    if not definition:
        return {}
    
    return {
        'name': definition.name,
        'description': definition.description,
        'bidirectional': definition.bidirectional,
        'weighted': definition.weight_range is not None,
        'weight_range': definition.weight_range,
        'inverse_type': definition.inverse_type
    }


def get_compatible_edge_types(source_type: str, target_type: str) -> List[str]:
    """Get edge types compatible with given node types."""
    return DEFAULT_SCHEMA.get_compatible_edges(source_type, target_type)


EDGE_SCHEMAS = {
    edge_type: {
        'source_types': defn.source_types,
        'target_types': defn.target_types,
        'weight_range': defn.weight_range,
        'bidirectional': defn.bidirectional,
        'description': defn.description,
        'inverse_type': defn.inverse_type
    }
    for edge_type, defn in EDGE_TYPE_DEFINITIONS.items()
}
