"""
Type definitions for GraphEngine.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, field
import uuid


class NodeType(str, Enum):
    """Supported node types in the knowledge graph."""
    CHARACTER = "character"
    SPECIES = "species"
    LOCATION = "location"
    FACTION = "faction"
    EVENT = "event"
    CONCEPT = "concept"
    PATTERN = "pattern"


class EdgeType(str, Enum):
    """Supported edge types in the knowledge graph."""
    # Ecological
    PREDATION = "predation"
    COMPETITION = "competition"
    MUTUALISM = "mutualism"
    
    # Narrative
    ALLIANCE = "alliance"
    CONFLICT = "conflict"
    DEPENDENCY = "dependency"
    
    # Spatial/Structural
    CONTAINS = "contains"
    ADJACENT = "adjacent"
    ORIGIN = "origin"
    
    # Semantic
    REFERENCES = "references"
    SIMILAR_TO = "similar_to"
    CAUSES = "causes"


class SourceType(str, Enum):
    """Source of node creation."""
    CA_GENERATED = "ca_generated"
    LLM_GENERATED = "llm_generated"
    USER_CREATED = "user_created"
    MODULE_CREATED = "module_created"


@dataclass
class RelationDef:
    """Definition of a relation to another node."""
    to: str
    type: str
    weight: float = 1.0
    context: str = ""
    bidirectional: bool = False


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "concept"
    created: datetime = field(default_factory=datetime.utcnow)
    modified: datetime = field(default_factory=datetime.utcnow)
    source: str = "user_created"
    embedding_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: List[RelationDef] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.type,
            'created': self.created.isoformat(),
            'modified': self.modified.isoformat(),
            'source': self.source,
            'embedding_id': self.embedding_id,
            'properties': self.properties,
            'relations': [
                {
                    'to': r.to,
                    'type': r.type,
                    'weight': r.weight,
                    'context': r.context,
                    'bidirectional': r.bidirectional
                }
                for r in self.relations
            ],
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeNode':
        """Create from dictionary."""
        relations = [
            RelationDef(
                to=r['to'],
                type=r['type'],
                weight=r.get('weight', 1.0),
                context=r.get('context', ''),
                bidirectional=r.get('bidirectional', False)
            )
            for r in data.get('relations', [])
        ]
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            type=data.get('type', 'concept'),
            created=datetime.fromisoformat(data['created']) if 'created' in data else datetime.utcnow(),
            modified=datetime.fromisoformat(data['modified']) if 'modified' in data else datetime.utcnow(),
            source=data.get('source', 'user_created'),
            embedding_id=data.get('embedding_id'),
            properties=data.get('properties', {}),
            relations=relations,
            tags=data.get('tags', [])
        )
    
    def get_name(self) -> str:
        """Get the display name for this node."""
        return self.properties.get('name', self.properties.get('title', self.id))


@dataclass
class KnowledgeEdge:
    """An edge in the knowledge graph."""
    id: int = 0
    source_id: str = ""
    target_id: str = ""
    edge_type: str = "references"
    weight: Optional[float] = None
    context: str = ""
    bidirectional: bool = False
    created: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type,
            'weight': self.weight,
            'context': self.context,
            'bidirectional': self.bidirectional,
            'created': self.created.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeEdge':
        """Create from dictionary."""
        return cls(
            id=data.get('id', 0),
            source_id=data.get('source_id', ''),
            target_id=data.get('target_id', ''),
            edge_type=data.get('edge_type', 'references'),
            weight=data.get('weight'),
            context=data.get('context', ''),
            bidirectional=data.get('bidirectional', False),
            created=datetime.fromisoformat(data['created']) if 'created' in data else datetime.utcnow()
        )


@dataclass
class ModuleResult:
    """Result from a processing module."""
    created_nodes: List[str] = field(default_factory=list)
    modified_nodes: List[str] = field(default_factory=list)
    created_edges: List[tuple] = field(default_factory=list)
    modified_edges: List[tuple] = field(default_factory=list)
    deleted_nodes: List[str] = field(default_factory=list)
    deleted_edges: List[tuple] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'created_nodes': self.created_nodes,
            'modified_nodes': self.modified_nodes,
            'created_edges': self.created_edges,
            'modified_edges': self.modified_edges,
            'deleted_nodes': self.deleted_nodes,
            'deleted_edges': self.deleted_edges,
            'metadata': self.metadata
        }
