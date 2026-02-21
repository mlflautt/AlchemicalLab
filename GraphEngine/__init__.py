"""
GraphEngine - Knowledge Graph System for AlchemicalLab
======================================================

A unified knowledge graph system that integrates CA-based generation
with Obsidian-compatible storage and BrainSimII-inspired processing modules.

Components:
- core: Node/edge schemas, main graph class
- storage: SQLite, ChromaDB, Obsidian sync
- modules: Processing units (evolution, narrative, etc.)
- bridges: Integration with CALab, StoryLab

Plan by: opencode/glm-5-free
Date: 2026-02-21
"""

from GraphEngine.core import (
    KnowledgeGraph,
    KnowledgeNode,
    KnowledgeEdge,
    RelationDef,
    ModuleResult,
    NodeType,
    EdgeType,
    SourceType,
    validate_node_properties,
    validate_edge,
)
from GraphEngine.core.node_schema import NodeSchema, NODE_TYPE_DEFINITIONS
from GraphEngine.core.edge_schema import EdgeSchema, EDGE_TYPE_DEFINITIONS

__all__ = [
    'KnowledgeGraph',
    'KnowledgeNode',
    'KnowledgeEdge',
    'RelationDef',
    'ModuleResult',
    'NodeType',
    'EdgeType',
    'SourceType',
    'NodeSchema',
    'EdgeSchema',
    'NODE_TYPE_DEFINITIONS',
    'EDGE_TYPE_DEFINITIONS',
    'validate_node_properties',
    'validate_edge',
]

__version__ = '0.1.0'
