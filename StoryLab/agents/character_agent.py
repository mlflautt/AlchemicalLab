"""
Character Agent Module for StoryLab.

Provides per-character simulation with limited perspective knowledge,
goal-directed decision making, and memory management.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random


class MemoryType(str, Enum):
    """Types of character memories."""
    EVENT = "event"
    CONVERSATION = "conversation"
    OBSERVATION = "observation"
    RELATIONSHIP = "relationship"
    GOAL = "goal"


@dataclass
class CharacterMemory:
    """A memory stored by a character."""
    memory_id: str
    memory_type: MemoryType
    content: str
    entities_involved: List[str]
    importance: float
    timestamp: str
    decay_rate: float = 0.01
    access_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'memory_id': self.memory_id,
            'memory_type': self.memory_type.value,
            'content': self.content,
            'entities_involved': self.entities_involved,
            'importance': self.importance,
            'timestamp': self.timestamp,
            'decay_rate': self.decay_rate,
            'access_count': self.access_count,
        }


@dataclass
class CharacterGoal:
    """A goal held by a character."""
    goal_id: str
    description: str
    priority: float
    status: str
    related_entities: List[str]
    progress: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'goal_id': self.goal_id,
            'description': self.description,
            'priority': self.priority,
            'status': self.status,
            'related_entities': self.related_entities,
            'progress': self.progress,
        }


@dataclass
class CharacterDecision:
    """A decision made by a character."""
    decision_id: str
    character_id: str
    action: str
    reasoning: str
    confidence: float
    affected_entities: List[str]
    expected_outcome: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'decision_id': self.decision_id,
            'character_id': self.character_id,
            'action': self.action,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'affected_entities': self.affected_entities,
            'expected_outcome': self.expected_outcome,
            'timestamp': self.timestamp,
        }


class CharacterAgent:
    """
    Agent representing a character's perspective and decision-making.
    
    Features:
    - Limited perspective knowledge (what the character knows)
    - Memory management with decay
    - Goal-directed behavior
    - Decision generation for LLM assistance
    """
    
    def __init__(
        self,
        graph: 'KnowledgeGraph',
        character_id: str,
        memory_limit: int = 50,
        llm_client: Any = None
    ):
        """
        Initialize character agent.
        
        Args:
            graph: KnowledgeGraph instance
            character_id: ID of character this agent represents
            memory_limit: Maximum memories to store
            llm_client: Optional LLM client for decision generation
        """
        self.graph = graph
        self.character_id = character_id
        self.memory_limit = memory_limit
        self.llm_client = llm_client
        
        self.memories: List[CharacterMemory] = []
        self.goals: List[CharacterGoal] = []
        self.known_entities: Set[str] = set()
        self.known_relationships: Dict[str, Dict] = {}
        
        self._initialize_from_graph()
    
    def _initialize_from_graph(self):
        """Initialize agent state from graph data."""
        character = self.graph.get_node(self.character_id)
        if not character:
            raise ValueError(f"Character {self.character_id} not found in graph")
        
        self.known_entities.add(self.character_id)
        
        neighbors = self.graph.get_neighbors(self.character_id, depth=1)
        for neighbor_id in neighbors:
            self.known_entities.add(neighbor_id)
        
        edges = self.graph.get_edges(self.character_id)
        for edge in edges:
            other_id = edge.target_id if edge.source_id == self.character_id else edge.source_id
            rel_type = edge.edge_type
            
            if other_id not in self.known_relationships:
                self.known_relationships[other_id] = {}
            self.known_relationships[other_id][rel_type] = {
                'weight': edge.weight,
                'context': edge.context,
            }
        
        goals_data = character.properties.get('goals', [])
        for i, goal_data in enumerate(goals_data):
            if isinstance(goal_data, str):
                self.goals.append(CharacterGoal(
                    goal_id=f"{self.character_id}_goal_{i}",
                    description=goal_data,
                    priority=0.5,
                    status='active',
                    related_entities=[],
                ))
    
    def get_knowledge(self) -> Dict[str, Any]:
        """
        Get what this character knows.
        
        Returns:
            Dict with known entities, relationships, and memories
        """
        known_entity_nodes = []
        for entity_id in self.known_entities:
            node = self.graph.get_node(entity_id)
            if node:
                known_entity_nodes.append({
                    'id': node.id,
                    'type': node.type,
                    'name': node.properties.get('name', node.id),
                })
        
        relevant_memories = self._get_relevant_memories()
        
        return {
            'character_id': self.character_id,
            'known_entities': known_entity_nodes,
            'known_relationships': self.known_relationships,
            'recent_memories': [m.to_dict() for m in relevant_memories[:10]],
            'active_goals': [g.to_dict() for g in self.goals if g.status == 'active'],
        }
    
    def observe(self, event_data: Dict[str, Any]) -> CharacterMemory:
        """
        Record an observation or event.
        
        Args:
            event_data: Dict with event_type, content, entities_involved, importance
        
        Returns:
            CharacterMemory created
        """
        memory = CharacterMemory(
            memory_id=f"{self.character_id}_mem_{datetime.utcnow().timestamp()}",
            memory_type=MemoryType(event_data.get('event_type', 'observation')),
            content=event_data.get('content', ''),
            entities_involved=event_data.get('entities_involved', []),
            importance=event_data.get('importance', 0.5),
            timestamp=datetime.utcnow().isoformat(),
        )
        
        for entity_id in event_data.get('entities_involved', []):
            self.known_entities.add(entity_id)
        
        self.memories.append(memory)
        
        if len(self.memories) > self.memory_limit:
            self._prune_memories()
        
        return memory
    
    def observe_conversation(
        self,
        other_character_id: str,
        topic: str,
        key_points: List[str]
    ) -> CharacterMemory:
        """Record a conversation with another character."""
        return self.observe({
            'event_type': 'conversation',
            'content': f"Talked with {other_character_id} about {topic}: {'; '.join(key_points)}",
            'entities_involved': [other_character_id],
            'importance': 0.6,
        })
    
    def observe_event(self, event_id: str, event_description: str) -> CharacterMemory:
        """Record witnessing an event."""
        event = self.graph.get_node(event_id)
        importance = event.properties.get('significance', 0.5) if event else 0.5
        
        return self.observe({
            'event_type': 'event',
            'content': event_description,
            'entities_involved': [event_id],
            'importance': importance,
        })
    
    def update_belief(self, entity_id: str, belief_updates: Dict[str, Any]):
        """
        Update beliefs about an entity.
        
        Args:
            entity_id: ID of entity to update beliefs about
            belief_updates: Dict of belief updates
        """
        if entity_id not in self.known_relationships:
            self.known_relationships[entity_id] = {}
        
        self.known_relationships[entity_id].update(belief_updates)
        
        self.known_entities.add(entity_id)
    
    def make_decision(
        self,
        situation: str,
        options: List[str] = None,
        context: Dict[str, Any] = None
    ) -> CharacterDecision:
        """
        Generate a decision in a situation.
        
        Args:
            situation: Description of the situation
            options: Optional list of possible actions
            context: Additional context
        
        Returns:
            CharacterDecision
        """
        knowledge = self.get_knowledge()
        
        reasoning = self._generate_reasoning(situation, knowledge, options)
        
        action = self._select_action(situation, options, reasoning)
        
        affected = self._determine_affected_entities(action)
        
        expected_outcome = self._predict_outcome(action, situation)
        
        decision = CharacterDecision(
            decision_id=f"{self.character_id}_dec_{datetime.utcnow().timestamp()}",
            character_id=self.character_id,
            action=action,
            reasoning=reasoning,
            confidence=self._calculate_confidence(action, reasoning),
            affected_entities=affected,
            expected_outcome=expected_outcome,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        self._store_decision_in_graph(decision)
        
        return decision
    
    def _generate_reasoning(
        self,
        situation: str,
        knowledge: Dict[str, Any],
        options: List[str]
    ) -> str:
        """Generate reasoning for decision."""
        character = self.graph.get_node(self.character_id)
        if not character:
            return "Unable to reason - character not found."
        
        motivations = character.properties.get('motivation', 'unknown goals')
        personality_traits = character.properties.get('traits', {})
        
        relevant_goals = [
            g for g in self.goals
            if g.status == 'active'
        ]
        
        reasoning_parts = [
            f"My motivation is: {motivations}",
        ]
        
        if relevant_goals:
            top_goal = max(relevant_goals, key=lambda g: g.priority)
            reasoning_parts.append(f"My current priority is: {top_goal.description}")
        
        if self.known_relationships:
            close_relations = [
                k for k, v in self.known_relationships.items()
                if v.get('weight', 0) > 0.7
            ]
            if close_relations:
                reasoning_parts.append(f"I have strong connections to: {', '.join(close_relations[:3])}")
        
        return " | ".join(reasoning_parts)
    
    def _select_action(
        self,
        situation: str,
        options: List[str],
        reasoning: str
    ) -> str:
        """Select an action based on situation and reasoning."""
        if options:
            if self.llm_client:
                return self._llm_select_action(situation, options, reasoning)
            else:
                scored_options = [
                    (opt, self._score_option(opt, reasoning))
                    for opt in options
                ]
                scored_options.sort(key=lambda x: x[1], reverse=True)
                return scored_options[0][0]
        
        return self._improvise_action(situation)
    
    def _score_option(self, option: str, reasoning: str) -> float:
        """Score an option based on character state."""
        score = 0.5
        
        for goal in self.goals:
            if goal.status == 'active':
                goal_keywords = goal.description.lower().split()
                if any(kw in option.lower() for kw in goal_keywords):
                    score += goal.priority * 0.3
        
        character = self.graph.get_node(self.character_id)
        if character:
            motivation = character.properties.get('motivation', '').lower()
            if any(word in option.lower() for word in motivation.split()):
                score += 0.2
        
        return min(1.0, score)
    
    def _improvise_action(self, situation: str) -> str:
        """Generate an improvised action when no options given."""
        character = self.graph.get_node(self.character_id)
        if not character:
            return "Wait and observe"
        
        role = character.properties.get('role', 'unknown')
        
        role_actions = {
            'protagonist': 'Take decisive action to address the situation',
            'antagonist': 'Create an obstacle or complication',
            'mentor': 'Offer guidance or wisdom',
            'ally': 'Support the protagonist',
            'unknown': 'Observe and gather information',
        }
        
        return role_actions.get(role, 'Act according to character nature')
    
    def _llm_select_action(
        self,
        situation: str,
        options: List[str],
        reasoning: str
    ) -> str:
        """Use LLM to select action (placeholder)."""
        return options[0] if options else "Undecided"
    
    def _determine_affected_entities(self, action: str) -> List[str]:
        """Determine which entities are affected by an action."""
        affected = []
        
        for entity_id in self.known_entities:
            node = self.graph.get_node(entity_id)
            if not node:
                continue
            
            name = node.properties.get('name', '').lower()
            if name and name in action.lower():
                affected.append(entity_id)
        
        return affected[:5]
    
    def _predict_outcome(self, action: str, situation: str) -> str:
        """Predict the outcome of an action."""
        return f"Expected outcome of '{action[:50]}...': to be determined by world simulation"
    
    def _calculate_confidence(self, action: str, reasoning: str) -> float:
        """Calculate confidence in decision."""
        confidence = 0.5
        
        if len(self.goals) > 0:
            confidence += 0.1
        
        if len(self.memories) > 5:
            confidence += 0.1
        
        if len(self.known_entities) > 3:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _store_decision_in_graph(self, decision: CharacterDecision):
        """Store decision as event in graph."""
        try:
            event_id = self.graph.add_node(
                node_type='event',
                properties={
                    'name': f"Decision by {self.character_id}",
                    'event_type': 'decision',
                    'description': decision.action,
                    'reasoning': decision.reasoning,
                    'confidence': decision.confidence,
                },
                tags=['decision', 'character_agent'],
                source='character_agent',
            )
            
            self.graph.add_edge(
                source_id=event_id,
                target_id=self.character_id,
                edge_type='causes',
                weight=decision.confidence,
                context='character_decision',
            )
            
            for entity_id in decision.affected_entities:
                self.graph.add_edge(
                    source_id=event_id,
                    target_id=entity_id,
                    edge_type='references',
                    context='affected_by_decision',
                )
        
        except Exception:
            pass
    
    def _get_relevant_memories(self, limit: int = 10) -> List[CharacterMemory]:
        """Get most relevant memories."""
        scored_memories = []
        
        for memory in self.memories:
            recency_score = self._recency_score(memory)
            importance_score = memory.importance
            access_score = min(1.0, memory.access_count / 10)
            
            relevance = (
                recency_score * 0.4 +
                importance_score * 0.4 +
                access_score * 0.2
            )
            
            scored_memories.append((memory, relevance))
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        selected = [m for m, _ in scored_memories[:limit]]
        
        for memory in selected:
            memory.access_count += 1
        
        return selected
    
    def _recency_score(self, memory: CharacterMemory) -> float:
        """Calculate recency score for a memory."""
        try:
            memory_time = datetime.fromisoformat(memory.timestamp)
            age_hours = (datetime.utcnow() - memory_time).total_seconds() / 3600
            
            if age_hours < 1:
                return 1.0
            elif age_hours < 24:
                return 0.8
            elif age_hours < 168:
                return 0.5
            else:
                return 0.2
        except:
            return 0.5
    
    def _prune_memories(self):
        """Remove least important memories."""
        if len(self.memories) <= self.memory_limit:
            return
        
        scored = [
            (m, m.importance * (1 - m.decay_rate * m.access_count))
            for m in self.memories
        ]
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        self.memories = [m for m, _ in scored[:self.memory_limit]]
    
    def add_goal(
        self,
        description: str,
        priority: float = 0.5,
        related_entities: List[str] = None
    ) -> CharacterGoal:
        """Add a new goal."""
        goal = CharacterGoal(
            goal_id=f"{self.character_id}_goal_{datetime.utcnow().timestamp()}",
            description=description,
            priority=priority,
            status='active',
            related_entities=related_entities or [],
        )
        
        self.goals.append(goal)
        
        return goal
    
    def update_goal_progress(self, goal_id: str, progress: float):
        """Update progress on a goal."""
        for goal in self.goals:
            if goal.goal_id == goal_id:
                goal.progress = progress
                if progress >= 1.0:
                    goal.status = 'completed'
                break
    
    def forget_entity(self, entity_id: str):
        """Remove knowledge of an entity."""
        self.known_entities.discard(entity_id)
        self.known_relationships.pop(entity_id, None)
        
        self.memories = [
            m for m in self.memories
            if entity_id not in m.entities_involved
        ]


class CharacterAgentManager:
    """
    Manages multiple character agents.
    """
    
    def __init__(self, graph: 'KnowledgeGraph', llm_client: Any = None):
        self.graph = graph
        self.llm_client = llm_client
        self.agents: Dict[str, CharacterAgent] = {}
    
    def get_agent(self, character_id: str) -> CharacterAgent:
        """Get or create agent for a character."""
        if character_id not in self.agents:
            self.agents[character_id] = CharacterAgent(
                graph=self.graph,
                character_id=character_id,
                llm_client=self.llm_client,
            )
        
        return self.agents[character_id]
    
    def broadcast_observation(
        self,
        event_data: Dict[str, Any],
        character_ids: List[str] = None
    ):
        """Broadcast an observation to multiple characters."""
        if character_ids is None:
            character_ids = [
                node.id for node in self.graph.list_nodes(node_type='character')
            ]
        
        for char_id in character_ids:
            if char_id in self.agents:
                self.agents[char_id].observe(event_data)
    
    def get_all_knowledge(self) -> Dict[str, Dict]:
        """Get knowledge state of all agents."""
        return {
            char_id: agent.get_knowledge()
            for char_id, agent in self.agents.items()
        }
    
    def sync_to_graph(self):
        """Sync agent knowledge back to graph."""
        for agent in self.agents.values():
            for entity_id in agent.known_entities:
                self.graph.add_edge(
                    source_id=agent.character_id,
                    target_id=entity_id,
                    edge_type='knows_of',
                    weight=0.5,
                )
