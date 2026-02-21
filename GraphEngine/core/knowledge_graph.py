"""
Main KnowledgeGraph class for GraphEngine.

Integrates SQLite storage, ChromaDB vector search, and Obsidian sync.
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import uuid

from GraphEngine.core.types import (
    KnowledgeNode, KnowledgeEdge, RelationDef, ModuleResult, NodeType, EdgeType, SourceType
)
from GraphEngine.core.node_schema import validate_node_properties, get_required_properties
from GraphEngine.core.edge_schema import validate_edge, get_edge_semantics
from GraphEngine.storage.sqlite_backend import SQLiteBackend
from GraphEngine.storage.chroma_bridge import ChromaBridge
from GraphEngine.storage.obsidian_sync import ObsidianSync


class KnowledgeGraph:
    """
    Main knowledge graph class integrating storage, search, and sync.
    
    Features:
    - SQLite for persistent node/edge storage
    - ChromaDB for vector similarity search
    - Obsidian vault bidirectional sync
    - Type-safe node and edge schemas
    - Module-based processing pipeline
    """
    
    def __init__(
        self,
        db_path: str = "knowledge_graph.db",
        chroma_path: str = "./chroma_db",
        vault_path: str = None,
        enable_vector_search: bool = True
    ):
        """
        Initialize KnowledgeGraph.
        
        Args:
            db_path: Path to SQLite database
            chroma_path: Path to ChromaDB persistence directory
            vault_path: Optional path to Obsidian vault
            enable_vector_search: Whether to enable ChromaDB vector search
        """
        self.db = SQLiteBackend(db_path)
        self.chroma = ChromaBridge(chroma_path) if enable_vector_search else None
        self.obsidian = ObsidianSync(vault_path, self) if vault_path else None
        
        self._modules: Dict[str, 'ProcessingModule'] = {}
        self._type_validators: Dict[str, callable] = {}
    
    def add_node(
        self,
        node_type: str,
        properties: Dict[str, Any],
        relations: List[RelationDef] = None,
        tags: List[str] = None,
        source: str = "user_created",
        node_id: str = None
    ) -> str:
        """
        Add a node to the graph.
        
        Args:
            node_type: Type of node (character, species, location, etc.)
            properties: Node properties
            relations: List of relations to other nodes
            tags: List of tags
            source: Source of node creation
            node_id: Optional explicit node ID
        
        Returns:
            Node ID
        """
        if relations is None:
            relations = []
        if tags is None:
            tags = []
        
        if node_id is None:
            node_id = str(uuid.uuid4())
        
        validate_node_properties(node_type, properties)
        
        node = KnowledgeNode(
            id=node_id,
            type=node_type,
            source=source,
            properties=properties,
            relations=relations,
            tags=tags
        )
        
        self.db.add_node(node)
        
        if self.chroma:
            embedding_id = self.chroma.index_node(node)
            if embedding_id:
                self.db.update_node(node_id, {'embedding_id': embedding_id})
        
        return node_id
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID."""
        return self.db.get_node(node_id)
    
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        node = self.db.get_node(node_id)
        if not node:
            return False
        
        merged = dict(node.properties)
        merged.update(properties)
        
        validate_node_properties(node.type, merged)
        
        success = self.db.update_node(node_id, properties)
        
        if success and self.chroma:
            updated_node = self.db.get_node(node_id)
            self.chroma.update_node(updated_node)
        
        return success
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        if self.chroma:
            self.chroma.delete_node(node_id)
        
        return self.db.delete_node(node_id)
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = None,
        context: str = "",
        bidirectional: bool = False
    ) -> int:
        """Add an edge between nodes."""
        validate_edge(source_id, target_id, edge_type)
        
        return self.db.add_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            context=context,
            bidirectional=bidirectional
        )
    
    def get_edges(
        self,
        node_id: str,
        edge_types: List[str] = None,
        direction: str = "both"
    ) -> List[KnowledgeEdge]:
        """Get edges for a node."""
        return self.db.get_edges(node_id, edge_types, direction)
    
    def delete_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        """Delete an edge."""
        return self.db.delete_edge(source_id, target_id, edge_type)
    
    def search(
        self,
        query: str,
        node_types: List[str] = None,
        source: str = None,
        use_vector: bool = True,
        limit: int = 10
    ) -> List[str]:
        """
        Search for nodes.
        
        Args:
            query: Search query
            node_types: Filter by node types
            source: Filter by source
            use_vector: Use vector search if available
            limit: Max results
        
        Returns:
            List of matching node IDs
        """
        if use_vector and self.chroma and self.chroma.available:
            return self.chroma.search_similar(query, node_types, source, limit)
        else:
            return self.db.text_search(query, node_types, limit)
    
    def get_neighbors(
        self,
        node_id: str,
        edge_types: List[str] = None,
        direction: str = "both",
        depth: int = 1
    ) -> List[str]:
        """Get neighboring node IDs."""
        return self.db.get_neighbors(node_id, edge_types, direction, depth)
    
    def list_node_ids(
        self,
        node_type: str = None,
        source: str = None,
        limit: int = None
    ) -> List[str]:
        """List all node IDs with optional filtering."""
        nodes = self.db.list_nodes(node_type, source, limit)
        return [n.id for n in nodes]
    
    def list_nodes(
        self,
        node_type: str = None,
        source: str = None,
        limit: int = None
    ) -> List[KnowledgeNode]:
        """List nodes with optional filtering."""
        return self.db.list_nodes(node_type, source, limit)
    
    def count_nodes(self, node_type: str = None) -> int:
        """Count nodes."""
        return self.db.count_nodes(node_type)
    
    def register_module(self, name: str, module: 'ProcessingModule'):
        """Register a processing module."""
        self._modules[name] = module
    
    def run_module(self, name: str, **kwargs) -> ModuleResult:
        """Run a registered module."""
        if name not in self._modules:
            raise ValueError(f"Module '{name}' not registered")
        
        module = self._modules[name]
        return module.process(self, **kwargs)
    
    def export_to_obsidian(self, node_id: str = None) -> List[str]:
        """Export node(s) to Obsidian vault."""
        if not self.obsidian:
            raise RuntimeError("Obsidian vault not configured")
        
        if node_id:
            path = self.obsidian.export_node(node_id)
            return [str(path)] if path else []
        else:
            return self.obsidian.export_all()
    
    def import_from_obsidian(self) -> List[str]:
        """Import all MD files from Obsidian vault."""
        if not self.obsidian:
            raise RuntimeError("Obsidian vault not configured")
        
        return self.obsidian.import_vault()
    
    def sync_obsidian(self) -> Dict[str, Any]:
        """Bidirectional sync with Obsidian vault."""
        if not self.obsidian:
            raise RuntimeError("Obsidian vault not configured")
        
        return self.obsidian.sync()
    
    def export_graph(self) -> Dict:
        """Export entire graph to dictionary."""
        return self.db.export_to_dict()
    
    def import_graph(self, data: Dict) -> tuple:
        """Import graph from dictionary."""
        return self.db.import_from_dict(data)
    
    def get_subgraph(
        self,
        node_ids: List[str],
        include_edges: bool = True
    ) -> Dict:
        """
        Extract a subgraph containing specified nodes.
        
        Args:
            node_ids: List of node IDs to include
            include_edges: Whether to include edges between nodes
        
        Returns:
            Dict with nodes and edges
        """
        nodes = []
        edges = []
        seen_edges = set()
        
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                nodes.append(node.to_dict())
                
                if include_edges:
                    for edge in self.get_edges(node_id):
                        if edge.target_id in node_ids:
                            edge_key = (edge.source_id, edge.target_id, edge.edge_type)
                            if edge_key not in seen_edges:
                                edges.append(edge.to_dict())
                                seen_edges.add(edge_key)
        
        return {'nodes': nodes, 'edges': edges}
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        edge_types: List[str] = None,
        max_depth: int = 5
    ) -> List[str]:
        """
        Find shortest path between two nodes.
        
        Returns:
            List of node IDs forming the path, or empty list if no path
        """
        if source_id == target_id:
            return [source_id]
        
        from collections import deque
        
        visited = {source_id}
        queue = deque([(source_id, [source_id])])
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            for neighbor in self.get_neighbors(current, edge_types, "both", 1):
                if neighbor == target_id:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def get_node_stats(self) -> Dict[str, int]:
        """Get statistics about nodes by type."""
        stats = {}
        for node_type in NodeType:
            count = self.count_nodes(node_type.value)
            if count > 0:
                stats[node_type.value] = count
        return stats
