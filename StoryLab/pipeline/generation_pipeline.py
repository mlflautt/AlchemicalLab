"""
Generation Pipeline for StoryLab.

Integrates GraphEngine context extraction with LLM generation.
Replaces the world_dna truncation approach with deterministic context extraction.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    from GraphEngine import KnowledgeGraph, ContextExtractor, ConsistencyChecker
    from GraphEngine.context import TaskType, ExtractedContext
    from GraphEngine.validation import ValidationResult, GeneratedContent
    GRAPH_ENGINE_AVAILABLE = True
except ImportError:
    GRAPH_ENGINE_AVAILABLE = False


@dataclass
class GenerationRequest:
    """A request for content generation."""
    task_type: str
    focus_entity_id: Optional[str] = None
    focus_entity_ids: Optional[List[str]] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 2000
    temperature: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of a generation request."""
    success: bool
    content: str
    context_used: Dict[str, Any]
    validation_result: Optional[Dict[str, Any]] = None
    entities_created: List[str] = field(default_factory=list)
    relationships_created: List[Tuple[str, str, str]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GenerationPipeline:
    """
    Pipeline for generating content using graph-based context.
    
    Flow:
    1. Extract context from KnowledgeGraph (deterministic, budget-aware)
    2. Format context for LLM prompt
    3. Generate content via LLM
    4. Parse and validate generated content
    5. Store valid entities/relationships in graph
    6. Return result with metadata
    """
    
    def __init__(
        self,
        graph: 'KnowledgeGraph',
        llm_client: Any = None,
        validate: bool = True,
        auto_store: bool = True
    ):
        """
        Initialize generation pipeline.
        
        Args:
            graph: KnowledgeGraph instance
            llm_client: LLM client (e.g., StoryLab's LLM interface)
            validate: Whether to validate generated content
            auto_store: Whether to automatically store valid content in graph
        """
        if not GRAPH_ENGINE_AVAILABLE:
            raise ImportError("GraphEngine not available. Install GraphEngine module.")
        
        self.graph = graph
        self.llm_client = llm_client
        self.validate = validate
        self.auto_store = auto_store
        
        self.context_extractor = ContextExtractor(graph)
        self.consistency_checker = ConsistencyChecker(graph) if validate else None
    
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate content based on request.
        
        Args:
            request: GenerationRequest with task type and focus entity
        
        Returns:
            GenerationResult with content and metadata
        """
        try:
            context = self._extract_context(request)
            
            prompt = self._build_prompt(request, context)
            
            content = self._call_llm(prompt, request)
            
            parsed = self._parse_content(content, request)
            
            validation_result = None
            if self.validate and parsed:
                validation_result = self.consistency_checker.validate(parsed)
            
            entities_created = []
            relationships_created = []
            
            if self.auto_store and parsed:
                if validation_result is None or validation_result.is_valid:
                    entities_created, relationships_created = self._store_in_graph(parsed)
            
            context_dict = context.to_dict() if hasattr(context, 'to_dict') else context
            
            return GenerationResult(
                success=True,
                content=content,
                context_used=context_dict,
                validation_result=validation_result.to_dict() if validation_result else None,
                entities_created=entities_created,
                relationships_created=relationships_created,
                metadata={
                    'task_type': request.task_type,
                    'focus_entity': request.focus_entity_id,
                    'timestamp': datetime.utcnow().isoformat(),
                }
            )
            
        except Exception as e:
            return GenerationResult(
                success=False,
                content="",
                context_used={},
                error=str(e),
                metadata={
                    'task_type': request.task_type,
                    'timestamp': datetime.utcnow().isoformat(),
                }
            )
    
    def generate_scene(
        self,
        location_id: str,
        character_ids: List[str],
        situation: str = "",
        max_tokens: int = 3000
    ) -> GenerationResult:
        """
        Generate a scene at a location with characters.
        
        Args:
            location_id: ID of location node
            character_ids: IDs of character nodes in scene
            situation: Optional situation description
            max_tokens: Max tokens for generation
        
        Returns:
            GenerationResult with scene content
        """
        request = GenerationRequest(
            task_type='scene_generation',
            focus_entity_id=location_id,
            focus_entity_ids=character_ids,
            additional_context={'situation': situation},
            max_tokens=max_tokens,
        )
        
        return self.generate(request)
    
    def generate_dialogue(
        self,
        speaker_id: str,
        listener_id: str,
        topic: str = "",
        max_tokens: int = 1500
    ) -> GenerationResult:
        """
        Generate dialogue between two characters.
        
        Args:
            speaker_id: ID of speaking character
            listener_id: ID of listening character
            topic: Optional topic for dialogue
            max_tokens: Max tokens for generation
        
        Returns:
            GenerationResult with dialogue content
        """
        request = GenerationRequest(
            task_type='dialogue',
            focus_entity_id=speaker_id,
            focus_entity_ids=[speaker_id, listener_id],
            additional_context={'topic': topic},
            max_tokens=max_tokens,
        )
        
        return self.generate(request)
    
    def generate_character_decision(
        self,
        character_id: str,
        situation: str,
        options: List[str] = None,
        max_tokens: int = 2000
    ) -> GenerationResult:
        """
        Generate a character's decision in a situation.
        
        Args:
            character_id: ID of character making decision
            situation: Description of situation
            options: Optional list of possible actions
            max_tokens: Max tokens for generation
        
        Returns:
            GenerationResult with decision content
        """
        request = GenerationRequest(
            task_type='character_decision',
            focus_entity_id=character_id,
            additional_context={
                'situation': situation,
                'options': options or [],
            },
            max_tokens=max_tokens,
        )
        
        return self.generate(request)
    
    def generate_world_event(
        self,
        trigger_description: str,
        affected_entity_ids: List[str] = None,
        max_tokens: int = 2500
    ) -> GenerationResult:
        """
        Generate a world event.
        
        Args:
            trigger_description: What triggers the event
            affected_entity_ids: IDs of entities affected
            max_tokens: Max tokens for generation
        
        Returns:
            GenerationResult with event content
        """
        request = GenerationRequest(
            task_type='world_event',
            focus_entity_ids=affected_entity_ids,
            additional_context={'trigger': trigger_description},
            max_tokens=max_tokens,
        )
        
        return self.generate(request)
    
    def _extract_context(self, request: GenerationRequest) -> 'ExtractedContext':
        """Extract context based on request type."""
        task_type = TaskType(request.task_type) if isinstance(request.task_type, str) else request.task_type
        
        if request.focus_entity_ids and len(request.focus_entity_ids) > 1:
            return self.context_extractor.get_context_for_characters(
                character_ids=request.focus_entity_ids,
                task_type=request.task_type,
                max_tokens=request.max_tokens
            )
        elif request.focus_entity_id:
            return self.context_extractor.get_context(
                entity_id=request.focus_entity_id,
                task_type=request.task_type,
                max_tokens=request.max_tokens
            )
        else:
            return self.context_extractor.get_context(
                entity_id=self._get_world_context_entity(),
                task_type=request.task_type,
                max_tokens=request.max_tokens
            )
    
    def _get_world_context_entity(self) -> str:
        """Get an entity ID for world-level context."""
        factions = self.graph.list_nodes(node_type='faction', limit=1)
        if factions:
            return factions[0].id
        
        locations = self.graph.list_nodes(node_type='location', limit=1)
        if locations:
            return locations[0].id
        
        characters = self.graph.list_nodes(node_type='character', limit=1)
        if characters:
            return characters[0].id
        
        raise ValueError("No entities in graph for context extraction")
    
    def _build_prompt(self, request: GenerationRequest, context: 'ExtractedContext') -> str:
        """Build LLM prompt from context."""
        context_text = context.serialize(format='markdown')
        
        task_prompts = {
            'scene_generation': self._scene_generation_prompt,
            'dialogue': self._dialogue_prompt,
            'character_decision': self._character_decision_prompt,
            'world_event': self._world_event_prompt,
            'faction_conflict': self._faction_conflict_prompt,
            'character_creation': self._character_creation_prompt,
            'location_description': self._location_description_prompt,
            'backstory_generation': self._backstory_prompt,
        }
        
        prompt_builder = task_prompts.get(request.task_type, self._generic_prompt)
        
        return prompt_builder(context_text, request)
    
    def _scene_generation_prompt(self, context: str, request: GenerationRequest) -> str:
        """Build prompt for scene generation."""
        situation = request.additional_context.get('situation', '')
        
        return f"""Generate a scene based on the following context.

{context}

{f"Situation: {situation}" if situation else ""}

Write a vivid scene that:
1. Reflects the atmosphere of the location
2. Shows character personalities through action and dialogue
3. Advances any active conflicts or tensions
4. Remains consistent with established relationships

Scene:"""
    
    def _dialogue_prompt(self, context: str, request: GenerationRequest) -> str:
        """Build prompt for dialogue generation."""
        topic = request.additional_context.get('topic', '')
        
        return f"""Generate dialogue between characters based on the following context.

{context}

{f"Topic: {topic}" if topic else ""}

Write natural dialogue that:
1. Reflects each character's personality and speech patterns
2. Considers their relationship (tension, alliance, etc.)
3. Reveals character through subtext
4. Remains consistent with shared history

Dialogue:"""
    
    def _character_decision_prompt(self, context: str, request: GenerationRequest) -> str:
        """Build prompt for character decision."""
        situation = request.additional_context.get('situation', '')
        options = request.additional_context.get('options', [])
        
        options_text = f"\nPossible actions:\n" + "\n".join(f"- {opt}" for opt in options) if options else ""
        
        return f"""Generate a character's decision based on the following context.

{context}

Situation: {situation}
{options_text}

Decide what this character would do, considering:
1. Their personality, goals, and motivations
2. Their knowledge of the situation (limited perspective)
3. Their relationships with involved parties
4. Past experiences that might influence the choice

Decision:"""
    
    def _world_event_prompt(self, context: str, request: GenerationRequest) -> str:
        """Build prompt for world event."""
        trigger = request.additional_context.get('trigger', '')
        
        return f"""Generate a world-changing event based on the following context.

{context}

Trigger: {trigger}

Create an event that:
1. Emerges naturally from existing tensions
2. Has meaningful consequences for affected entities
3. Respects established power dynamics
4. Opens new narrative possibilities

Event:"""
    
    def _faction_conflict_prompt(self, context: str, request: GenerationRequest) -> str:
        """Build prompt for faction conflict."""
        return f"""Generate faction conflict dynamics based on the following context.

{context}

Describe the conflict dynamics including:
1. The core issue driving the conflict
2. Each faction's goals and strategies
3. Key members involved
4. Potential outcomes and consequences

Conflict:"""
    
    def _character_creation_prompt(self, context: str, request: GenerationRequest) -> str:
        """Build prompt for character creation."""
        return f"""Create a new character that fits the world context.

{context}

Create a character that:
1. Fills a gap in the current cast (needed role)
2. Has clear connections to existing factions/locations
3. Has compelling motivations tied to world conflicts
4. Brings a unique perspective to the story

Character (JSON format with: name, role, motivation, appearance, backstory, faction_affiliations):"""
    
    def _location_description_prompt(self, context: str, request: GenerationRequest) -> str:
        """Build prompt for location description."""
        return f"""Generate a location description based on the following context.

{context}

Describe this location including:
1. Physical appearance and atmosphere
2. Notable features and landmarks
3. Typical inhabitants and activities
4. Historical significance

Description:"""
    
    def _backstory_prompt(self, context: str, request: GenerationRequest) -> str:
        """Build prompt for backstory generation."""
        return f"""Generate a backstory based on the following context.

{context}

Create a backstory that:
1. Explains the entity's current situation
2. Connects to existing world events and locations
3. Establishes key formative experiences
4. Provides hooks for future story development

Backstory:"""
    
    def _generic_prompt(self, context: str, request: GenerationRequest) -> str:
        """Generic prompt builder."""
        return f"""Generate content based on the following context.

{context}

Task: {request.task_type}

Content:"""
    
    def _call_llm(self, prompt: str, request: GenerationRequest) -> str:
        """Call LLM with prompt."""
        if self.llm_client is None:
            return self._mock_llm_response(prompt, request)
        
        if hasattr(self.llm_client, 'generate'):
            return self.llm_client.generate(prompt, temperature=request.temperature)
        
        if hasattr(self.llm_client, '__call__'):
            return self.llm_client(prompt, temperature=request.temperature)
        
        return self._mock_llm_response(prompt, request)
    
    def _mock_llm_response(self, prompt: str, request: GenerationRequest) -> str:
        """Generate mock response when no LLM available."""
        return f"[Mock generation for {request.task_type}]\n\nContext was extracted successfully. Connect an LLM client for real generation."
    
    def _parse_content(self, content: str, request: GenerationRequest) -> Optional['GeneratedContent']:
        """Parse generated content into structured data."""
        entities = []
        relationships = []
        events = []
        
        if request.task_type == 'character_creation':
            try:
                char_data = json.loads(content)
                entities.append({
                    'id': None,
                    'type': 'character',
                    'properties': char_data,
                })
            except json.JSONDecodeError:
                pass
        
        elif request.task_type == 'world_event':
            events.append({
                'id': None,
                'type': 'event',
                'properties': {
                    'name': request.additional_context.get('trigger', 'Generated Event'),
                    'description': content[:500],
                    'event_type': 'generated',
                },
            })
        
        return GeneratedContent(
            content_type=request.task_type,
            entities=entities,
            relationships=relationships,
            events=events,
            metadata={'raw_content': content[:1000]},
        )
    
    def _store_in_graph(
        self,
        content: 'GeneratedContent'
    ) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Store generated content in the graph."""
        entities_created = []
        relationships_created = []
        
        for entity_data in content.entities:
            try:
                entity_id = self.graph.add_node(
                    node_type=entity_data.get('type', 'concept'),
                    properties=entity_data.get('properties', {}),
                    tags=['generated', content.content_type],
                    source='llm_generated',
                )
                entities_created.append(entity_id)
            except Exception:
                pass
        
        for event_data in content.events:
            try:
                event_id = self.graph.add_node(
                    node_type='event',
                    properties=event_data.get('properties', {}),
                    tags=['generated', 'event'],
                    source='llm_generated',
                )
                entities_created.append(event_id)
            except Exception:
                pass
        
        for rel_data in content.relationships:
            try:
                self.graph.add_edge(
                    source_id=rel_data['source_id'],
                    target_id=rel_data['target_id'],
                    edge_type=rel_data.get('edge_type', 'references'),
                    weight=rel_data.get('weight'),
                    context=rel_data.get('context', ''),
                )
                relationships_created.append((
                    rel_data['source_id'],
                    rel_data['target_id'],
                    rel_data.get('edge_type', 'references'),
                ))
            except Exception:
                pass
        
        return entities_created, relationships_created
