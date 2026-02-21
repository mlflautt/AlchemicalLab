"""
SQLite Backend for GraphEngine.

Provides persistent storage for nodes and edges using SQLite.
"""

import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from contextlib import contextmanager
import os

from GraphEngine.core.types import KnowledgeNode, KnowledgeEdge, RelationDef


class SQLiteBackend:
    """SQLite-based storage backend for the knowledge graph."""
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize SQLite backend.
        
        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory
        """
        self.db_path = db_path
        self._ensure_schema()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with context management."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _ensure_schema(self):
        """Create database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Check and set schema version
            cursor.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            )
            row = cursor.fetchone()
            
            if row is None:
                cursor.execute(
                    "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
                    (str(self.SCHEMA_VERSION),)
                )
            
            # Nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT DEFAULT 'user_created',
                    embedding_id TEXT,
                    properties JSON,
                    tags JSON
                )
            """)
            
            # Edges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL,
                    context TEXT,
                    bidirectional BOOLEAN DEFAULT 0,
                    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    UNIQUE(source_id, target_id, edge_type)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type)")
            
            # Full-text search virtual table
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                    id,
                    type,
                    properties,
                    tags,
                    content='nodes',
                    content_rowid='rowid'
                )
            """)
    
    # Node Operations
    
    def add_node(self, node: KnowledgeNode) -> str:
        """Add a node to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO nodes (id, type, created, modified, source, embedding_id, properties, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.id,
                node.type,
                node.created.isoformat(),
                node.modified.isoformat(),
                node.source,
                node.embedding_id,
                json.dumps(node.properties),
                json.dumps(node.tags)
            ))
            
            # Add to FTS index
            cursor.execute("""
                INSERT INTO nodes_fts (id, type, properties, tags)
                VALUES (?, ?, ?, ?)
            """, (
                node.id,
                node.type,
                json.dumps(node.properties),
                json.dumps(node.tags)
            ))
            
            # Add relations as edges
            for rel in node.relations:
                cursor.execute("""
                    INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, weight, context, bidirectional)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    node.id,
                    rel.to,
                    rel.type,
                    rel.weight,
                    rel.context,
                    1 if rel.bidirectional else 0
                ))
            
            return node.id
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve a node by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            # Get relations
            cursor.execute(
                "SELECT target_id, edge_type, weight, context, bidirectional FROM edges WHERE source_id = ?",
                (node_id,)
            )
            relations = [
                RelationDef(
                    to=r['target_id'],
                    type=r['edge_type'],
                    weight=r['weight'] or 1.0,
                    context=r['context'] or '',
                    bidirectional=bool(r['bidirectional'])
                )
                for r in cursor.fetchall()
            ]
            
            return KnowledgeNode(
                id=row['id'],
                type=row['type'],
                created=datetime.fromisoformat(row['created']),
                modified=datetime.fromisoformat(row['modified']),
                source=row['source'],
                embedding_id=row['embedding_id'],
                properties=json.loads(row['properties']) if row['properties'] else {},
                relations=relations,
                tags=json.loads(row['tags']) if row['tags'] else []
            )
    
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get existing properties
            cursor.execute("SELECT properties FROM nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            
            if row is None:
                return False
            
            existing = json.loads(row['properties']) if row['properties'] else {}
            existing.update(properties)
            
            cursor.execute("""
                UPDATE nodes 
                SET properties = ?, modified = ?
                WHERE id = ?
            """, (
                json.dumps(existing),
                datetime.utcnow().isoformat(),
                node_id
            ))
            
            # Update FTS
            cursor.execute("""
                UPDATE nodes_fts 
                SET properties = ?
                WHERE id = ?
            """, (
                json.dumps(existing),
                node_id
            ))
            
            return True
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Remove from FTS
            cursor.execute("DELETE FROM nodes_fts WHERE id = ?", (node_id,))
            
            # Delete node (edges cascade via foreign keys)
            cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            
            return cursor.rowcount > 0
    
    def list_nodes(
        self,
        node_type: str = None,
        source: str = None,
        limit: int = None,
        offset: int = 0
    ) -> List[KnowledgeNode]:
        """List nodes with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT id FROM nodes WHERE 1=1"
            params = []
            
            if node_type:
                query += " AND type = ?"
                params.append(node_type)
            
            if source:
                query += " AND source = ?"
                params.append(source)
            
            query += " ORDER BY created DESC"
            
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            
            cursor.execute(query, params)
            
            return [self.get_node(row['id']) for row in cursor.fetchall()]
    
    def count_nodes(self, node_type: str = None) -> int:
        """Count nodes with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if node_type:
                cursor.execute("SELECT COUNT(*) FROM nodes WHERE type = ?", (node_type,))
            else:
                cursor.execute("SELECT COUNT(*) FROM nodes")
            
            return cursor.fetchone()[0]
    
    # Edge Operations
    
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
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO edges 
                (source_id, target_id, edge_type, weight, context, bidirectional)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                source_id,
                target_id,
                edge_type,
                weight,
                context,
                1 if bidirectional else 0
            ))
            
            return cursor.lastrowid
    
    def get_edges(
        self,
        node_id: str,
        edge_types: List[str] = None,
        direction: str = "both"
    ) -> List[KnowledgeEdge]:
        """Get edges for a node."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            queries = []
            params = []
            
            if direction in ("out", "both"):
                query = "SELECT * FROM edges WHERE source_id = ?"
                params.append(node_id)
                if edge_types:
                    query += f" AND edge_type IN ({','.join(['?'] * len(edge_types))})"
                    params.extend(edge_types)
                queries.append((query, params.copy()))
            
            if direction in ("in", "both"):
                params = [node_id]
                query = "SELECT * FROM edges WHERE target_id = ?"
                if edge_types:
                    query += f" AND edge_type IN ({','.join(['?'] * len(edge_types))})"
                    params.extend(edge_types)
                queries.append((query, params.copy()))
            
            edges = []
            for query, params in queries:
                cursor.execute(query, params)
                for row in cursor.fetchall():
                    edges.append(KnowledgeEdge(
                        id=row['id'],
                        source_id=row['source_id'],
                        target_id=row['target_id'],
                        edge_type=row['edge_type'],
                        weight=row['weight'],
                        context=row['context'] or "",
                        bidirectional=bool(row['bidirectional']),
                        created=datetime.fromisoformat(row['created'])
                    ))
            
            return edges
    
    def delete_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        """Delete a specific edge."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM edges 
                WHERE source_id = ? AND target_id = ? AND edge_type = ?
            """, (source_id, target_id, edge_type))
            
            return cursor.rowcount > 0
    
    # Search Operations
    
    def text_search(
        self,
        query: str,
        node_types: List[str] = None,
        limit: int = 10
    ) -> List[str]:
        """Full-text search on nodes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # FTS search
            fts_query = f"""
                SELECT id FROM nodes_fts 
                WHERE nodes_fts MATCH ? 
                ORDER BY rank
                LIMIT ?
            """
            cursor.execute(fts_query, (query, limit * 2))
            
            results = []
            for row in cursor.fetchall():
                node = self.get_node(row['id'])
                if node:
                    if node_types and node.type not in node_types:
                        continue
                    results.append(node.id)
                    if len(results) >= limit:
                        break
            
            return results
    
    # Graph Operations
    
    def get_neighbors(
        self,
        node_id: str,
        edge_types: List[str] = None,
        direction: str = "both",
        depth: int = 1
    ) -> List[str]:
        """Get neighboring node IDs."""
        if depth < 1:
            return []
        
        visited = {node_id}
        current_level = {node_id}
        all_neighbors = set()
        
        for _ in range(depth):
            next_level = set()
            
            for nid in current_level:
                edges = self.get_edges(nid, edge_types, direction)
                
                for edge in edges:
                    neighbor = edge.target_id if edge.source_id == nid else edge.source_id
                    if neighbor not in visited:
                        all_neighbors.add(neighbor)
                        next_level.add(neighbor)
                        visited.add(neighbor)
            
            current_level = next_level
        
        return list(all_neighbors)
    
    # Import/Export
    
    def export_to_dict(self) -> Dict:
        """Export entire graph to dictionary."""
        nodes = self.list_nodes(limit=100000)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM edges")
            edges = [dict(row) for row in cursor.fetchall()]
        
        return {
            'nodes': [n.to_dict() for n in nodes],
            'edges': edges,
            'metadata': {
                'exported': datetime.utcnow().isoformat(),
                'node_count': len(nodes),
                'edge_count': len(edges)
            }
        }
    
    def import_from_dict(self, data: Dict) -> Tuple[int, int]:
        """Import graph from dictionary."""
        node_count = 0
        edge_count = 0
        
        for node_data in data.get('nodes', []):
            node = KnowledgeNode.from_dict(node_data)
            self.add_node(node)
            node_count += 1
        
        for edge_data in data.get('edges', []):
            self.add_edge(
                source_id=edge_data['source_id'],
                target_id=edge_data['target_id'],
                edge_type=edge_data['edge_type'],
                weight=edge_data.get('weight'),
                context=edge_data.get('context', ''),
                bidirectional=edge_data.get('bidirectional', False)
            )
            edge_count += 1
        
        return node_count, edge_count
