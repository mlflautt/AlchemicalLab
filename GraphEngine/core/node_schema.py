"""
Node Schema Definitions for GraphEngine.

Defines node types, validation rules, and property schemas.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field


@dataclass
class NodeTypeDefinition:
    """Definition of a node type."""
    name: str
    required_properties: List[str] = field(default_factory=list)
    optional_properties: List[str] = field(default_factory=list)
    property_types: Dict[str, type] = field(default_factory=dict)
    property_defaults: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    icon: str = ""  # Obsidian-compatible icon


# Built-in node type definitions
NODE_TYPE_DEFINITIONS: Dict[str, NodeTypeDefinition] = {
    'character': NodeTypeDefinition(
        name='character',
        required_properties=['name', 'role'],
        optional_properties=[
            'motivation', 'appearance', 'backstory', 'secrets',
            'class', 'flaw', 'age', 'gender', 'occupation'
        ],
        property_types={
            'name': str,
            'role': str,
            'motivation': str,
            'appearance': str,
            'backstory': str,
            'secrets': str,
            'class': str,
            'flaw': str,
            'age': int,
            'gender': str,
            'occupation': str,
        },
        property_defaults={
            'role': 'Unknown',
        },
        description='A character in the narrative world',
        icon='👤'
    ),
    
    'species': NodeTypeDefinition(
        name='species',
        required_properties=['name', 'species_type'],
        optional_properties=[
            'population', 'traits', 'fitness', 'preferred_biomes',
            'energy_efficiency', 'reproduction_rate', 'dispersal_range',
            'center', 'territory_size'
        ],
        property_types={
            'name': str,
            'species_type': str,
            'population': int,
            'traits': dict,
            'fitness': float,
            'preferred_biomes': list,
            'energy_efficiency': float,
            'reproduction_rate': float,
            'dispersal_range': int,
            'center': tuple,
            'territory_size': int,
        },
        property_defaults={
            'species_type': 'unknown',
            'population': 100,
            'fitness': 0.5,
        },
        description='A biological species from ecosystem evolution',
        icon='🦎'
    ),
    
    'location': NodeTypeDefinition(
        name='location',
        required_properties=['name', 'location_type'],
        optional_properties=[
            'atmosphere', 'resources', 'coordinates', 'biome',
            'temperature', 'moisture', 'productivity', 'description'
        ],
        property_types={
            'name': str,
            'location_type': str,
            'atmosphere': str,
            'resources': str,
            'coordinates': tuple,
            'biome': str,
            'temperature': float,
            'moisture': float,
            'productivity': float,
            'description': str,
        },
        property_defaults={
            'location_type': 'unknown',
            'atmosphere': 'neutral',
        },
        description='A location in the world',
        icon='📍'
    ),
    
    'faction': NodeTypeDefinition(
        name='faction',
        required_properties=['name', 'ideology'],
        optional_properties=[
            'methods', 'goals', 'members', 'territories',
            'power_level', 'resources', 'description'
        ],
        property_types={
            'name': str,
            'ideology': str,
            'methods': str,
            'goals': str,
            'members': list,
            'territories': list,
            'power_level': float,
            'resources': dict,
            'description': str,
        },
        property_defaults={
            'ideology': 'neutral',
            'power_level': 0.5,
        },
        description='A political or social faction',
        icon='🏛️'
    ),
    
    'event': NodeTypeDefinition(
        name='event',
        required_properties=['name', 'event_type'],
        optional_properties=[
            'timestamp', 'participants', 'outcomes', 'significance',
            'location', 'duration', 'description'
        ],
        property_types={
            'name': str,
            'event_type': str,
            'timestamp': str,
            'participants': list,
            'outcomes': list,
            'significance': float,
            'location': str,
            'duration': str,
            'description': str,
        },
        property_defaults={
            'event_type': 'unknown',
            'significance': 0.5,
        },
        description='An event in the world history',
        icon='⚡'
    ),
    
    'concept': NodeTypeDefinition(
        name='concept',
        required_properties=['name', 'concept_type'],
        optional_properties=[
            'definition', 'examples', 'related_concepts',
            'source', 'domain'
        ],
        property_types={
            'name': str,
            'concept_type': str,
            'definition': str,
            'examples': list,
            'related_concepts': list,
            'source': str,
            'domain': str,
        },
        property_defaults={
            'concept_type': 'abstract',
        },
        description='An abstract concept or idea',
        icon='💡'
    ),
    
    'pattern': NodeTypeDefinition(
        name='pattern',
        required_properties=['pattern_type', 'source_system'],
        optional_properties=[
            'parameters', 'state_vector', 'generation',
            'size', 'period', 'center', 'trajectory'
        ],
        property_types={
            'pattern_type': str,
            'source_system': str,
            'parameters': dict,
            'state_vector': list,
            'generation': int,
            'size': int,
            'period': int,
            'center': tuple,
            'trajectory': list,
        },
        property_defaults={
            'source_system': 'unknown',
            'generation': 0,
        },
        description='A pattern detected in CA or other systems',
        icon='🔷'
    ),
}


class NodeSchema:
    """Schema validation and management for nodes."""
    
    def __init__(self, custom_definitions: Optional[Dict[str, NodeTypeDefinition]] = None):
        self.definitions = NODE_TYPE_DEFINITIONS.copy()
        if custom_definitions:
            self.definitions.update(custom_definitions)
    
    def get_definition(self, node_type: str) -> Optional[NodeTypeDefinition]:
        """Get the definition for a node type."""
        return self.definitions.get(node_type)
    
    def list_types(self) -> List[str]:
        """List all known node types."""
        return list(self.definitions.keys())
    
    def validate(
        self, 
        node_type: str, 
        properties: Dict[str, Any]
    ) -> tuple:
        """
        Validate properties against node type schema.
        
        Returns:
            (is_valid, errors, normalized_properties)
        """
        definition = self.definitions.get(node_type)
        
        if not definition:
            return False, [f"Unknown node type: {node_type}"], properties
        
        errors = []
        normalized = {}
        
        # Check required properties
        for prop in definition.required_properties:
            if prop not in properties:
                if prop in definition.property_defaults:
                    normalized[prop] = definition.property_defaults[prop]
                else:
                    errors.append(f"Missing required property: {prop}")
            else:
                normalized[prop] = properties[prop]
        
        # Add optional properties with defaults
        for prop in definition.optional_properties:
            if prop in properties:
                normalized[prop] = properties[prop]
            elif prop in definition.property_defaults:
                normalized[prop] = definition.property_defaults[prop]
        
        # Validate types
        for prop, value in normalized.items():
            if prop in definition.property_types:
                expected_type = definition.property_types[prop]
                if not isinstance(value, expected_type) and value is not None:
                    errors.append(
                        f"Property '{prop}' should be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
        
        return len(errors) == 0, errors, normalized
    
    def create_template(self, node_type: str) -> Dict[str, Any]:
        """Create a template dictionary for a node type."""
        definition = self.definitions.get(node_type)
        if not definition:
            return {}
        
        template = {}
        
        for prop in definition.required_properties:
            if prop in definition.property_defaults:
                template[prop] = definition.property_defaults[prop]
            else:
                template[prop] = ""
        
        for prop in definition.optional_properties:
            if prop in definition.property_defaults:
                template[prop] = definition.property_defaults[prop]
        
        return template
    
    def register_type(self, definition: NodeTypeDefinition):
        """Register a new node type definition."""
        self.definitions[definition.name] = definition


DEFAULT_SCHEMA = NodeSchema()


def validate_node_properties(node_type: str, properties: Dict[str, Any]) -> tuple:
    """
    Validate node properties against schema.
    
    Args:
        node_type: Type of node
        properties: Properties to validate
    
    Returns:
        (is_valid, errors) tuple
    """
    is_valid, errors, _ = DEFAULT_SCHEMA.validate(node_type, properties)
    return is_valid, errors


def get_required_properties(node_type: str) -> List[str]:
    """Get required properties for a node type."""
    definition = NODE_TYPE_DEFINITIONS.get(node_type)
    return definition.required_properties if definition else []


def get_optional_properties(node_type: str) -> List[str]:
    """Get optional properties for a node type."""
    definition = NODE_TYPE_DEFINITIONS.get(node_type)
    return definition.optional_properties if definition else []


NODE_TYPE_PROPERTIES = {
    node_type: {
        'required': defn.required_properties,
        'optional': defn.optional_properties,
        'description': defn.description,
        'icon': defn.icon
    }
    for node_type, defn in NODE_TYPE_DEFINITIONS.items()
}
