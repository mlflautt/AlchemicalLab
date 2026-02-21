"""
StoryLab Bridge for GraphEngine.

Provides integration between StoryLab LLM generation system
and the knowledge graph.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

from GraphEngine.core.types import ModuleResult, RelationDef
from GraphEngine.modules.narrative_generation import NarrativeGenerationModule


class StoryLabBridge:
    """
    Bridge between StoryLab and GraphEngine.
    
    Converts StoryLab generated content (characters, plots, worldbuilding)
    into knowledge graph nodes and edges.
    """
    
    def __init__(self, graph: 'KnowledgeGraph'):
        self.graph = graph
        
        self.narrative_module = NarrativeGenerationModule()
        graph.register_module('storylab_narrative', self.narrative_module)
    
    def process_generated_content(
        self,
        content_type: str,
        content: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Process generated content from StoryLab.
        
        Args:
            content_type: Type of content ('character', 'plot', 'worldbuilding', 'scene')
            content: Generated content dict
            metadata: Optional metadata about generation
        
        Returns:
            Node ID of created/updated node, or None if failed
        """
        if metadata is None:
            metadata = {}
        
        processors = {
            'character': self._process_character,
            'plot': self._process_plot,
            'worldbuilding': self._process_worldbuilding,
            'scene': self._process_scene,
            'faction': self._process_faction
        }
        
        processor = processors.get(content_type)
        if not processor:
            return None
        
        return processor(content, metadata)
    
    def _process_character(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Process a generated character."""
        name = content.get('name', 'Unknown')
        char_id = content.get('id') or hashlib.md5(name.encode()).hexdigest()[:12]
        
        properties = {
            'name': name,
            'role': content.get('role', 'NPC'),
            'motivation': content.get('motivation', ''),
            'appearance': content.get('appearance', ''),
            'backstory': content.get('backstory', ''),
            'secrets': content.get('secrets', ''),
            'flaw': content.get('flaw', ''),
            'importance': content.get('importance', 0.5)
        }
        
        tags = ['character', 'generated', 'storylab']
        if content.get('is_protagonist'):
            tags.append('protagonist')
        if content.get('is_antagonist'):
            tags.append('antagonist')
        
        source = 'llm_generated'
        if metadata.get('model'):
            source = f"llm_{metadata['model']}"
        
        existing = self.graph.get_node(char_id)
        if existing:
            self.graph.update_node(char_id, properties)
            return char_id
        
        return self.graph.add_node(
            node_type='character',
            properties=properties,
            tags=tags,
            source=source,
            node_id=char_id
        )
    
    def _process_plot(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Process a generated plot structure."""
        title = content.get('title', 'Untitled Plot')
        plot_id = content.get('id') or hashlib.md5(title.encode()).hexdigest()[:12]
        
        properties = {
            'name': title,
            'event_type': 'plot',
            'description': content.get('description', ''),
            'beats': content.get('beats', []),
            'climax': content.get('climax', ''),
            'resolution': content.get('resolution', ''),
            'themes': content.get('themes', [])
        }
        
        node_id = self.graph.add_node(
            node_type='event',
            properties=properties,
            tags=['plot', 'generated', 'storylab'],
            source='llm_generated',
            node_id=plot_id
        )
        
        if 'characters' in content:
            for char_name in content['characters']:
                char_nodes = self.graph.search(char_name, node_types=['character'], limit=1)
                if char_nodes:
                    self.graph.add_edge(
                        source_id=plot_id,
                        target_id=char_nodes[0],
                        edge_type='references',
                        context='involved_in_plot'
                    )
        
        return node_id
    
    def _process_worldbuilding(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Process worldbuilding content."""
        world_type = content.get('type', 'general')
        
        if world_type == 'location':
            return self._process_location(content, metadata)
        elif world_type == 'faction':
            return self._process_faction(content, metadata)
        elif world_type == 'concept':
            return self._process_concept(content, metadata)
        else:
            return self._process_generic_worldbuilding(content, metadata)
    
    def _process_location(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Process a generated location."""
        name = content.get('name', 'Unknown Location')
        loc_id = content.get('id') or hashlib.md5(name.encode()).hexdigest()[:12]
        
        properties = {
            'name': name,
            'location_type': content.get('location_type', 'unknown'),
            'atmosphere': content.get('atmosphere', 'neutral'),
            'description': content.get('description', ''),
            'resources': content.get('resources', '')
        }
        
        return self.graph.add_node(
            node_type='location',
            properties=properties,
            tags=['location', 'generated', 'storylab'],
            source='llm_generated',
            node_id=loc_id
        )
    
    def _process_faction(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Process a generated faction."""
        name = content.get('name', 'Unknown Faction')
        fac_id = content.get('id') or hashlib.md5(name.encode()).hexdigest()[:12]
        
        properties = {
            'name': name,
            'ideology': content.get('ideology', 'neutral'),
            'methods': content.get('methods', ''),
            'goals': content.get('goals', ''),
            'description': content.get('description', ''),
            'power_level': content.get('power_level', 0.5)
        }
        
        return self.graph.add_node(
            node_type='faction',
            properties=properties,
            tags=['faction', 'generated', 'storylab'],
            source='llm_generated',
            node_id=fac_id
        )
    
    def _process_concept(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Process a generated concept/theme."""
        name = content.get('name', 'Unknown Concept')
        concept_id = content.get('id') or hashlib.md5(name.encode()).hexdigest()[:12]
        
        properties = {
            'name': name,
            'concept_type': content.get('concept_type', 'abstract'),
            'definition': content.get('definition', ''),
            'examples': content.get('examples', []),
            'source': content.get('source', 'storylab')
        }
        
        return self.graph.add_node(
            node_type='concept',
            properties=properties,
            tags=['concept', 'generated', 'storylab'],
            source='llm_generated',
            node_id=concept_id
        )
    
    def _process_generic_worldbuilding(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Process generic worldbuilding content."""
        name = content.get('name', content.get('title', 'Unknown'))
        node_id = content.get('id') or hashlib.md5(name.encode()).hexdigest()[:12]
        
        properties = dict(content)
        properties['name'] = name
        
        return self.graph.add_node(
            node_type='concept',
            properties=properties,
            tags=['worldbuilding', 'generated', 'storylab'],
            source='llm_generated',
            node_id=node_id
        )
    
    def _process_scene(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Process a generated scene."""
        title = content.get('title', f"Scene at {content.get('location', 'Unknown')}")
        scene_id = content.get('id') or hashlib.md5(title.encode()).hexdigest()[:12]
        
        properties = {
            'name': title,
            'event_type': 'scene',
            'description': content.get('description', ''),
            'location': content.get('location', ''),
            'participants': content.get('participants', []),
            'conflict': content.get('conflict', ''),
            'outcome': content.get('outcome', ''),
            'emotional_beat': content.get('emotional_beat', '')
        }
        
        node_id = self.graph.add_node(
            node_type='event',
            properties=properties,
            tags=['scene', 'generated', 'storylab'],
            source='llm_generated',
            node_id=scene_id
        )
        
        if 'characters' in content:
            for char_name in content['characters']:
                char_nodes = self.graph.search(char_name, node_types=['character'], limit=1)
                if char_nodes:
                    self.graph.add_edge(
                        source_id=scene_id,
                        target_id=char_nodes[0],
                        edge_type='references',
                        context='participant'
                    )
        
        return node_id
    
    def create_character_relationships(
        self,
        character_id: str,
        relationships: List[Dict[str, Any]]
    ) -> List[tuple]:
        """
        Create relationships for a character.
        
        Args:
            character_id: ID of the character node
            relationships: List of relationship dicts
        
        Returns:
            List of created edge tuples
        """
        created = []
        
        for rel in relationships:
            target_name = rel.get('target')
            rel_type = rel.get('type', 'references')
            weight = rel.get('weight', 0.5)
            context = rel.get('context', '')
            
            target_nodes = self.graph.search(target_name, node_types=['character'], limit=1)
            if target_nodes:
                edge = self.graph.add_edge(
                    source_id=character_id,
                    target_id=target_nodes[0],
                    edge_type=rel_type,
                    weight=weight,
                    context=context,
                    bidirectional=rel.get('bidirectional', False)
                )
                created.append((character_id, target_nodes[0], rel_type))
        
        return created
    
    def get_story_context(self, character_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get story context for a character.
        
        Args:
            character_id: ID of the character
            depth: How many relationship levels to traverse
        
        Returns:
            Dict with related characters, events, and locations
        """
        char_node = self.graph.get_node(character_id)
        if not char_node:
            return {}
        
        neighbor_ids = self.graph.get_neighbors(character_id, depth=depth)
        neighbors = [self.graph.get_node(nid) for nid in neighbor_ids]
        neighbors = [n for n in neighbors if n]
        
        context = {
            'character': char_node.to_dict(),
            'related_characters': [],
            'related_events': [],
            'related_locations': [],
            'related_factions': [],
            'other': []
        }
        
        for neighbor in neighbors:
            n_dict = neighbor.to_dict()
            n_type = neighbor.type
            
            if n_type == 'character':
                context['related_characters'].append(n_dict)
            elif n_type == 'event':
                context['related_events'].append(n_dict)
            elif n_type == 'location':
                context['related_locations'].append(n_dict)
            elif n_type == 'faction':
                context['related_factions'].append(n_dict)
            else:
                context['other'].append(n_dict)
        
        return context
    
    def export_for_generation(self) -> Dict[str, Any]:
        """
        Export graph data for LLM generation context.
        
        Returns:
            Dict with summarized graph data suitable for prompting
        """
        characters = self.graph.list_nodes(node_type='character', limit=20)
        factions = self.graph.list_nodes(node_type='faction', limit=10)
        locations = self.graph.list_nodes(node_type='location', limit=10)
        events = self.graph.list_nodes(node_type='event', limit=15)
        
        return {
            'characters': [
                {
                    'name': c.get_name(),
                    'role': c.properties.get('role', 'unknown'),
                    'motivation': c.properties.get('motivation', '')
                }
                for c in characters
            ],
            'factions': [
                {
                    'name': f.get_name(),
                    'ideology': f.properties.get('ideology', 'unknown')
                }
                for f in factions
            ],
            'locations': [
                {
                    'name': l.get_name(),
                    'type': l.properties.get('location_type', 'unknown'),
                    'atmosphere': l.properties.get('atmosphere', 'neutral')
                }
                for l in locations
            ],
            'recent_events': [
                {
                    'name': e.get_name(),
                    'type': e.properties.get('event_type', 'unknown')
                }
                for e in events
            ]
        }
