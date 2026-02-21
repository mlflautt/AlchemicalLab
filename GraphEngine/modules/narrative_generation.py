"""
Narrative Generation Processing Module.

Processes narrative elements and generates story-related content.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

from GraphEngine.core.types import ModuleResult, RelationDef
from GraphEngine.modules.base_module import ProcessingModule


class NarrativeGenerationModule(ProcessingModule):
    """
    Processing module for narrative generation.
    
    Takes CA narrative patterns and LLM outputs to:
    - Create/update character nodes
    - Generate faction relationships
    - Track plot events and arcs
    - Create thematic connections
    """
    
    def __init__(self):
        super().__init__(
            name="narrative_generation",
            description="Process narrative elements into characters, factions, and plot events"
        )
        
        self.config = {
            'min_character_importance': 0.3,
            'max_faction_members': 50,
            'track_story_arcs': True,
            'generate_conflicts': True
        }
    
    def process(self, graph: 'KnowledgeGraph', **kwargs) -> ModuleResult:
        """
        Process narrative data into the knowledge graph.
        
        Args:
            graph: KnowledgeGraph instance
            narrative_state: Dict with narrative data
            world_dna: Optional world DNA from CA
            generation: Current generation/iteration
        
        Returns:
            ModuleResult with created/modified nodes and edges
        """
        result = ModuleResult()
        
        narrative_state = kwargs.get('narrative_state', {})
        world_dna = kwargs.get('world_dna', {})
        generation = kwargs.get('generation', 0)
        
        if not narrative_state and not world_dna:
            return result
        
        if 'characters' in narrative_state:
            for char in narrative_state['characters']:
                node_id = self._process_character(graph, char, generation)
                if node_id:
                    result.created_nodes.append(node_id)
        
        if 'factions' in narrative_state:
            for faction in narrative_state['factions']:
                node_id = self._process_faction(graph, faction, generation)
                if node_id:
                    result.created_nodes.append(node_id)
        
        if 'events' in narrative_state:
            for event in narrative_state['events']:
                node_id = self._process_event(graph, event, generation)
                if node_id:
                    result.created_nodes.append(node_id)
        
        if world_dna:
            self._process_world_dna(graph, world_dna, result)
        
        if self.config['generate_conflicts']:
            conflicts = self._generate_conflicts(graph, narrative_state)
            for conflict in conflicts:
                edge_key = self._create_conflict_edge(graph, conflict)
                if edge_key:
                    result.created_edges.append(edge_key)
        
        result.metadata = {
            'generation': generation,
            'characters_processed': len(narrative_state.get('characters', [])),
            'factions_processed': len(narrative_state.get('factions', [])),
            'events_processed': len(narrative_state.get('events', [])),
            'conflicts_generated': len(result.created_edges)
        }
        
        return result
    
    def _process_character(
        self,
        graph: 'KnowledgeGraph',
        character: Dict[str, Any],
        generation: int
    ) -> Optional[str]:
        """Create or update a character node."""
        char_id = character.get('id')
        if not char_id:
            name = character.get('name', 'Unknown')
            char_id = hashlib.md5(name.encode()).hexdigest()[:12]
        
        properties = {
            'name': character.get('name', 'Unknown'),
            'role': character.get('role', 'NPC'),
            'motivation': character.get('motivation', ''),
            'importance': character.get('importance', 0.5)
        }
        
        for optional_prop in ['appearance', 'backstory', 'secrets', 'flaw', 'class', 'age', 'gender', 'occupation']:
            if optional_prop in character:
                properties[optional_prop] = character[optional_prop]
        
        existing = graph.get_node(char_id)
        
        if existing:
            graph.update_node(char_id, properties)
            return char_id
        else:
            tags = ['character', 'narrative']
            if character.get('is_protagonist'):
                tags.append('protagonist')
            if character.get('is_antagonist'):
                tags.append('antagonist')
            
            return graph.add_node(
                node_type='character',
                properties=properties,
                tags=tags,
                source='llm_generated',
                node_id=char_id
            )
    
    def _process_faction(
        self,
        graph: 'KnowledgeGraph',
        faction: Dict[str, Any],
        generation: int
    ) -> Optional[str]:
        """Create or update a faction node."""
        faction_id = faction.get('id')
        if not faction_id:
            name = faction.get('name', 'Unknown Faction')
            faction_id = hashlib.md5(name.encode()).hexdigest()[:12]
        
        properties = {
            'name': faction.get('name', 'Unknown Faction'),
            'ideology': faction.get('ideology', 'neutral'),
            'power_level': faction.get('power_level', 0.5),
            'methods': faction.get('methods', ''),
            'goals': faction.get('goals', '')
        }
        
        if 'description' in faction:
            properties['description'] = faction['description']
        if 'resources' in faction:
            properties['resources'] = faction['resources']
        
        existing = graph.get_node(faction_id)
        
        if existing:
            graph.update_node(faction_id, properties)
            node_id = faction_id
        else:
            node_id = graph.add_node(
                node_type='faction',
                properties=properties,
                tags=['faction', 'narrative'],
                source='llm_generated',
                node_id=faction_id
            )
        
        if 'members' in faction:
            for member_name in faction['members'][:self.config['max_faction_members']]:
                member_nodes = graph.search(member_name, node_types=['character'], limit=1)
                if member_nodes:
                    graph.add_edge(
                        source_id=faction_id,
                        target_id=member_nodes[0],
                        edge_type='contains',
                        context='member'
                    )
        
        return node_id
    
    def _process_event(
        self,
        graph: 'KnowledgeGraph',
        event: Dict[str, Any],
        generation: int
    ) -> Optional[str]:
        """Create or update an event node."""
        event_id = event.get('id')
        if not event_id:
            name = event.get('name', 'Unknown Event')
            timestamp = event.get('timestamp', str(generation))
            event_id = hashlib.md5(f"{name}_{timestamp}".encode()).hexdigest()[:12]
        
        properties = {
            'name': event.get('name', 'Unknown Event'),
            'event_type': event.get('event_type', 'plot'),
            'timestamp': event.get('timestamp', str(generation)),
            'significance': event.get('significance', 0.5),
            'description': event.get('description', '')
        }
        
        if 'outcomes' in event:
            properties['outcomes'] = event['outcomes']
        if 'location' in event:
            properties['location'] = event['location']
        
        existing = graph.get_node(event_id)
        
        if existing:
            graph.update_node(event_id, properties)
            return event_id
        else:
            return graph.add_node(
                node_type='event',
                properties=properties,
                tags=['event', 'narrative', event.get('event_type', 'plot')],
                source='llm_generated',
                node_id=event_id
            )
    
    def _process_world_dna(
        self,
        graph: 'KnowledgeGraph',
        world_dna: Dict[str, Any],
        result: ModuleResult
    ):
        """Process world DNA from CA narrative bridge."""
        themes = world_dna.get('themes', [])
        for theme in themes:
            node_id = graph.add_node(
                node_type='concept',
                properties={
                    'name': theme.get('name', 'Unknown Theme'),
                    'concept_type': 'theme',
                    'definition': theme.get('description', ''),
                    'source': 'ca_world_dna'
                },
                tags=['theme', 'world-building'],
                source='ca_generated'
            )
            if node_id:
                result.created_nodes.append(node_id)
        
        patterns = world_dna.get('narrative_patterns', [])
        for pattern in patterns:
            node_id = graph.add_node(
                node_type='pattern',
                properties={
                    'name': pattern.get('name', 'Unknown Pattern'),
                    'pattern_type': pattern.get('type', 'narrative'),
                    'source_system': 'ca',
                    'parameters': pattern.get('parameters', {})
                },
                tags=['pattern', 'narrative'],
                source='ca_generated'
            )
            if node_id:
                result.created_nodes.append(node_id)
    
    def _generate_conflicts(
        self,
        graph: 'KnowledgeGraph',
        narrative_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate conflict relationships between entities."""
        conflicts = []
        
        characters = narrative_state.get('characters', [])
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                if self._should_conflict(char1, char2):
                    conflicts.append({
                        'type': 'character_conflict',
                        'source': char1.get('id'),
                        'target': char2.get('id'),
                        'weight': 0.7,
                        'context': f"{char1.get('role', '')} vs {char2.get('role', '')}"
                    })
        
        factions = narrative_state.get('factions', [])
        for i, fac1 in enumerate(factions):
            for fac2 in factions[i+1:]:
                if self._should_faction_conflict(fac1, fac2):
                    conflicts.append({
                        'type': 'faction_conflict',
                        'source': fac1.get('id'),
                        'target': fac2.get('id'),
                        'weight': 0.8,
                        'context': f"{fac1.get('ideology', '')} vs {fac2.get('ideology', '')}"
                    })
        
        return conflicts
    
    def _should_conflict(self, char1: Dict, char2: Dict) -> bool:
        """Determine if two characters should be in conflict."""
        if char1.get('is_protagonist') and char2.get('is_antagonist'):
            return True
        if char1.get('is_antagonist') and char2.get('is_protagonist'):
            return True
        
        ideologies = []
        for char in [char1, char2]:
            if 'ideology' in char:
                ideologies.append(char['ideology'])
        
        if len(ideologies) == 2 and ideologies[0] != ideologies[1]:
            return True
        
        return False
    
    def _should_faction_conflict(self, fac1: Dict, fac2: Dict) -> bool:
        """Determine if two factions should be in conflict."""
        ideologies = [fac1.get('ideology'), fac2.get('ideology')]
        
        opposing_pairs = [
            ('order', 'chaos'),
            ('good', 'evil'),
            ('conservative', 'progressive'),
            ('expansionist', 'isolationist')
        ]
        
        for opp1, opp2 in opposing_pairs:
            if opp1 in ideologies and opp2 in ideologies:
                return True
        
        return False
    
    def _create_conflict_edge(
        self,
        graph: 'KnowledgeGraph',
        conflict: Dict[str, Any]
    ) -> Optional[tuple]:
        """Create a conflict edge in the graph."""
        source = conflict.get('source')
        target = conflict.get('target')
        
        if not source or not target:
            return None
        
        if not graph.get_node(source) or not graph.get_node(target):
            return None
        
        graph.add_edge(
            source_id=source,
            target_id=target,
            edge_type='conflict',
            weight=conflict.get('weight', 0.5),
            context=conflict.get('context', ''),
            bidirectional=False
        )
        
        return (source, target, 'conflict')
