"""
GraphEngine Core module.

Provides the main KnowledgeGraph class and type definitions.
"""

from GraphEngine.core.types import (
    NodeType, EdgeType, SourceType,
    KnowledgeNode, KnowledgeEdge, RelationDef, ModuleResult
)
from GraphEngine.core.node_schema import (
    validate_node_properties,
    get_required_properties,
    NODE_TYPE_PROPERTIES
)
from GraphEngine.core.edge_schema import (
    validate_edge,
    get_edge_semantics,
    EDGE_SCHEMAS
)
from GraphEngine.core.knowledge_graph import KnowledgeGraph

__all__ = [
    'NodeType', 'EdgeType', 'SourceType',
    'KnowledgeNode', 'KnowledgeEdge', 'RelationDef', 'ModuleResult',
    'validate_node_properties', 'get_required_properties', 'NODE_TYPE_PROPERTIES',
    'validate_edge', 'get_edge_semantics', 'EDGE_SCHEMAS',
    'KnowledgeGraph'
]
