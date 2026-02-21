"""
ChromaDB Bridge for GraphEngine.

Provides vector search capabilities for nodes.
"""

from typing import Dict, List, Optional, Any
import json

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class ChromaBridge:
    """Bridge between KnowledgeGraph and ChromaDB for vector search."""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "knowledge_nodes",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize ChromaDB bridge.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
            embedding_model: Name of sentence-transformers model
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        self._client = None
        self._collection = None
        self._embedder = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        if not CHROMADB_AVAILABLE:
            print("Warning: chromadb not available. Vector search disabled.")
            return
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: sentence-transformers not available. Using ChromaDB's default embedding.")
            self._embedder = None
        else:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    @property
    def available(self) -> bool:
        """Check if ChromaDB is available."""
        return self._client is not None and self._collection is not None
    
    def _node_to_text(self, node) -> str:
        """Convert node to searchable text."""
        parts = []
        
        # Add type
        parts.append(f"Type: {node.type}")
        
        # Add properties
        for key, value in node.properties.items():
            if value is not None:
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                parts.append(f"{key}: {value}")
        
        # Add tags
        if node.tags:
            parts.append(f"Tags: {', '.join(node.tags)}")
        
        return "\n".join(parts)
    
    def index_node(self, node) -> Optional[str]:
        """Add node to vector index."""
        if not self.available:
            return None
        
        text = self._node_to_text(node)
        
        if self._embedder:
            embedding = self._embedder.encode(text).tolist()
            self._collection.add(
                ids=[node.id],
                embeddings=[embedding],
                metadatas=[{
                    "type": node.type,
                    "source": node.source,
                    "created": node.created.isoformat(),
                    "name": node.properties.get("name", "")
                }],
                documents=[text]
            )
        else:
            self._collection.add(
                ids=[node.id],
                metadatas=[{
                    "type": node.type,
                    "source": node.source,
                    "created": node.created.isoformat(),
                    "name": node.properties.get("name", "")
                }],
                documents=[text]
            )
        
        return node.id
    
    def update_node(self, node) -> Optional[str]:
        """Update node in vector index."""
        if not self.available:
            return None
        
        # ChromaDB doesn't have update, so we delete and re-add
        self.delete_node(node.id)
        return self.index_node(node)
    
    def delete_node(self, node_id: str) -> bool:
        """Remove node from vector index."""
        if not self.available:
            return False
        
        try:
            self._collection.delete(ids=[node_id])
            return True
        except Exception:
            return False
    
    def search_similar(
        self,
        query: str,
        node_types: List[str] = None,
        source: str = None,
        k: int = 10
    ) -> List[str]:
        """
        Find similar nodes by semantic search.
        
        Args:
            query: Search query text
            node_types: Filter by node types
            source: Filter by source
            k: Number of results to return
        
        Returns:
            List of node IDs
        """
        if not self.available:
            return []
        
        # Build where filter
        where_filter = None
        if node_types or source:
            conditions = []
            if node_types:
                conditions.append({"type": {"$in": node_types}})
            if source:
                conditions.append({"source": source})
            
            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}
        
        # Generate query embedding if using custom embedder
        if self._embedder:
            query_embedding = self._embedder.encode(query).tolist()
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter
            )
        else:
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter
            )
        
        return results['ids'][0] if results['ids'] else []
    
    def get_embeddings(self, node_ids: List[str]) -> Dict[str, List[float]]:
        """Get embeddings for specific nodes."""
        if not self.available:
            return {}
        
        try:
            results = self._collection.get(
                ids=node_ids,
                include=["embeddings"]
            )
            
            return {
                node_id: embedding
                for node_id, embedding in zip(results['ids'], results['embeddings'] or [])
            }
        except Exception:
            return {}
    
    def count(self) -> int:
        """Count indexed nodes."""
        if not self.available:
            return 0
        
        return self._collection.count()
    
    def clear(self):
        """Clear all indexed nodes."""
        if not self.available:
            return
        
        # Get all IDs and delete
        results = self._collection.get()
        if results['ids']:
            self._collection.delete(ids=results['ids'])
