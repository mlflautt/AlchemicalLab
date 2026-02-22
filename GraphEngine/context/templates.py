"""
Context Templates for GraphEngine.

Defines task-specific context schemas and extraction rules.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    """Types of generation tasks."""
    SCENE_GENERATION = "scene_generation"
    DIALOGUE = "dialogue"
    CHARACTER_DECISION = "character_decision"
    FACTION_CONFLICT = "faction_conflict"
    WORLD_EVENT = "world_event"
    CHARACTER_CREATION = "character_creation"
    LOCATION_DESCRIPTION = "location_description"
    NARRATIVE_SUMMARY = "narrative_summary"
    CONFLICT_RESOLUTION = "conflict_resolution"
    BACKSTORY_GENERATION = "backstory_generation"


@dataclass
class ContextRequirement:
    """Requirement for context extraction."""
    node_types: List[str]
    edge_types: List[str]
    max_depth: int
    max_nodes: int
    importance_threshold: float = 0.0
    include_temporal: bool = False
    include_conflicts: bool = False
    include_causal_chain: bool = False


@dataclass
class TaskTemplate:
    """Template for a specific generation task."""
    task_type: TaskType
    description: str
    focus_entity_types: List[str]
    requirements: ContextRequirement
    prompt_sections: List[str]
    max_tokens: int
    priority_ordering: List[str] = field(default_factory=list)
    
    def get_section_order(self) -> List[str]:
        """Get ordered list of prompt sections."""
        if self.priority_ordering:
            return self.priority_ordering
        return [
            'focus_entity',
            'relationships',
            'related_entities',
            'temporal_context',
            'conflicts',
            'world_state',
        ]


TASK_TEMPLATES: Dict[TaskType, TaskTemplate] = {
    TaskType.SCENE_GENERATION: TaskTemplate(
        task_type=TaskType.SCENE_GENERATION,
        description="Generate a scene involving characters at a location",
        focus_entity_types=['character', 'location'],
        requirements=ContextRequirement(
            node_types=['character', 'location', 'event', 'faction'],
            edge_types=['alliance', 'conflict', 'contains', 'references', 'origin'],
            max_depth=2,
            max_nodes=15,
            importance_threshold=0.1,
            include_temporal=True,
            include_conflicts=True,
        ),
        prompt_sections=[
            'location_context',
            'present_characters',
            'character_relationships',
            'recent_events',
            'active_conflicts',
            'scene_directions',
        ],
        max_tokens=2500,
        priority_ordering=[
            'focus_entity',
            'related_entities',
            'relationships',
            'temporal_context',
            'conflicts',
        ],
    ),
    
    TaskType.DIALOGUE: TaskTemplate(
        task_type=TaskType.DIALOGUE,
        description="Generate dialogue between two or more characters",
        focus_entity_types=['character'],
        requirements=ContextRequirement(
            node_types=['character', 'event', 'faction'],
            edge_types=['alliance', 'conflict', 'references'],
            max_depth=1,
            max_nodes=5,
            importance_threshold=0.2,
            include_temporal=True,
            include_conflicts=False,
        ),
        prompt_sections=[
            'speakers',
            'relationship_context',
            'shared_history',
            'current_situation',
            'dialogue_tone',
        ],
        max_tokens=1000,
    ),
    
    TaskType.CHARACTER_DECISION: TaskTemplate(
        task_type=TaskType.CHARACTER_DECISION,
        description="Generate a character's decision in a situation",
        focus_entity_types=['character'],
        requirements=ContextRequirement(
            node_types=['character', 'location', 'faction', 'event'],
            edge_types=['alliance', 'conflict', 'dependency', 'contains'],
            max_depth=2,
            max_nodes=10,
            importance_threshold=0.15,
            include_temporal=True,
            include_conflicts=True,
        ),
        prompt_sections=[
            'character_state',
            'available_knowledge',
            'goals_motivations',
            'constraints',
            'stakeholders',
        ],
        max_tokens=1500,
    ),
    
    TaskType.FACTION_CONFLICT: TaskTemplate(
        task_type=TaskType.FACTION_CONFLICT,
        description="Generate faction conflict dynamics",
        focus_entity_types=['faction'],
        requirements=ContextRequirement(
            node_types=['faction', 'character', 'location', 'event'],
            edge_types=['alliance', 'conflict', 'contains', 'competition'],
            max_depth=2,
            max_nodes=20,
            importance_threshold=0.1,
            include_temporal=True,
            include_conflicts=True,
            include_causal_chain=True,
        ),
        prompt_sections=[
            'factions_involved',
            'key_members',
            'territories',
            'conflict_history',
            'resources',
            'stakes',
        ],
        max_tokens=3000,
    ),
    
    TaskType.WORLD_EVENT: TaskTemplate(
        task_type=TaskType.WORLD_EVENT,
        description="Generate a world-changing event",
        focus_entity_types=['location', 'faction', 'character'],
        requirements=ContextRequirement(
            node_types=['character', 'faction', 'location', 'event', 'species'],
            edge_types=['alliance', 'conflict', 'contains', 'causes', 'references'],
            max_depth=3,
            max_nodes=25,
            importance_threshold=0.05,
            include_temporal=True,
            include_conflicts=True,
            include_causal_chain=True,
        ),
        prompt_sections=[
            'affected_entities',
            'causal_context',
            'location_context',
            'timing',
            'potential_consequences',
        ],
        max_tokens=2500,
    ),
    
    TaskType.CHARACTER_CREATION: TaskTemplate(
        task_type=TaskType.CHARACTER_CREATION,
        description="Create a new character fitting the world",
        focus_entity_types=['faction', 'location'],
        requirements=ContextRequirement(
            node_types=['faction', 'location', 'character', 'concept'],
            edge_types=['alliance', 'conflict', 'contains'],
            max_depth=2,
            max_nodes=30,
            importance_threshold=0.0,
            include_temporal=False,
            include_conflicts=True,
        ),
        prompt_sections=[
            'world_context',
            'existing_factions',
            'existing_characters',
            'locations',
            'themes',
            'gaps_to_fill',
        ],
        max_tokens=3500,
    ),
    
    TaskType.LOCATION_DESCRIPTION: TaskTemplate(
        task_type=TaskType.LOCATION_DESCRIPTION,
        description="Generate a location description",
        focus_entity_types=['location'],
        requirements=ContextRequirement(
            node_types=['location', 'character', 'event', 'species'],
            edge_types=['contains', 'origin', 'references', 'adjacent'],
            max_depth=1,
            max_nodes=10,
            importance_threshold=0.1,
            include_temporal=True,
            include_conflicts=False,
        ),
        prompt_sections=[
            'location_properties',
            'inhabitants',
            'recent_events',
            'connections',
            'atmosphere',
        ],
        max_tokens=1000,
    ),
    
    TaskType.NARRATIVE_SUMMARY: TaskTemplate(
        task_type=TaskType.NARRATIVE_SUMMARY,
        description="Generate a summary of narrative state",
        focus_entity_types=['character', 'faction', 'event'],
        requirements=ContextRequirement(
            node_types=['character', 'faction', 'event', 'location', 'concept'],
            edge_types=['alliance', 'conflict', 'causes', 'references'],
            max_depth=2,
            max_nodes=40,
            importance_threshold=0.2,
            include_temporal=True,
            include_conflicts=True,
        ),
        prompt_sections=[
            'key_characters',
            'active_arcs',
            'major_events',
            'world_state',
            'pending_conflicts',
        ],
        max_tokens=4000,
    ),
    
    TaskType.CONFLICT_RESOLUTION: TaskTemplate(
        task_type=TaskType.CONFLICT_RESOLUTION,
        description="Generate resolution for a conflict",
        focus_entity_types=['character', 'faction'],
        requirements=ContextRequirement(
            node_types=['character', 'faction', 'event', 'location'],
            edge_types=['alliance', 'conflict', 'causes', 'dependency'],
            max_depth=2,
            max_nodes=15,
            importance_threshold=0.15,
            include_temporal=True,
            include_conflicts=True,
            include_causal_chain=True,
        ),
        prompt_sections=[
            'conflicting_parties',
            'conflict_causes',
            'stakeholders',
            'possible_resolutions',
            'consequences',
        ],
        max_tokens=2000,
    ),
    
    TaskType.BACKSTORY_GENERATION: TaskTemplate(
        task_type=TaskType.BACKSTORY_GENERATION,
        description="Generate backstory for a character or faction",
        focus_entity_types=['character', 'faction'],
        requirements=ContextRequirement(
            node_types=['character', 'faction', 'location', 'event'],
            edge_types=['origin', 'alliance', 'conflict', 'contains'],
            max_depth=2,
            max_nodes=12,
            importance_threshold=0.1,
            include_temporal=True,
            include_conflicts=False,
        ),
        prompt_sections=[
            'entity_properties',
            'origin_location',
            'formative_events',
            'key_relationships',
            'motivations',
        ],
        max_tokens=1500,
    ),
}


def get_template(task_type: TaskType) -> TaskTemplate:
    """Get template for a task type."""
    return TASK_TEMPLATES.get(task_type)


def get_requirements(task_type: TaskType) -> ContextRequirement:
    """Get context requirements for a task type."""
    template = get_template(task_type)
    return template.requirements if template else None


def list_task_types() -> List[str]:
    """List all available task types."""
    return [t.value for t in TaskType]


def estimate_tokens(node_count: int, edge_count: int, include_temporal: bool = False) -> int:
    """Estimate token count for context."""
    base_tokens = 50
    node_tokens = node_count * 30
    edge_tokens = edge_count * 15
    temporal_tokens = 100 if include_temporal else 0
    
    return base_tokens + node_tokens + edge_tokens + temporal_tokens
