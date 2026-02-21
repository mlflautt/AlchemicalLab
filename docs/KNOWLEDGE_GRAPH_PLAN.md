# Knowledge Graph Architecture Plan

**Model**: opencode/glm-5-free  
**Date**: 2026-02-21  
**Status**: Implementation Phase

---

## Executive Summary

This document outlines the architecture for a unified Knowledge Graph system that integrates AlchemicalLab's CA-based generation with Obsidian-compatible storage and BrainSimII-inspired processing modules.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH CORE                             │
│  (Obsidian-compatible, typed nodes/edges, versioned)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│   │  Nodes   │    │  Edges   │    │ Modules  │    │  Views   │   │
│   │(Entities)│    │(Relations)│    │(Process) │    │(Visual)  │   │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘   │
│        │               │               │               │          │
│        └───────────────┴───────────────┴───────────────┘          │
│                              │                                      │
│                    ┌─────────┴─────────┐                           │
│                    │  Graph Database   │                           │
│                    │  (SQLite/Neo4j)   │                           │
│                    └───────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│    CALab      │     │   StoryLab    │     │   SynthLab    │
│               │     │               │     │               │
│ species nodes │     │ story nodes   │     │ CA state nodes│
│ biome nodes   │     │ character n.  │     │ agent nodes   │
│ eco edges     │     │ narrative e.  │     │ semantic edges│
└───────────────┘     └───────────────┘     └───────────────┘
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Graph Storage | SQLite | Lightweight, file-based, transactional |
| Vector Search | ChromaDB | Semantic similarity, embeddings |
| File Format | Obsidian MD | Human-readable, tool-compatible |
| Processing | Python Modules | BrainSimII-inspired processing units |
| Visualization | D3.js + Obsidian | Web view + native graph view |

---

## Node Schema

### Node Types

```yaml
# Core entity types
node_types:
  character:
    required: [name, role]
    optional: [motivation, appearance, backstory, secrets]
    
  species:
    required: [name, species_type]
    optional: [population, traits, fitness, preferred_biomes]
    
  location:
    required: [name, location_type]
    optional: [atmosphere, resources, coordinates, biome]
    
  faction:
    required: [name, ideology]
    optional: [methods, goals, members, territories]
    
  event:
    required: [name, event_type, timestamp]
    optional: [participants, outcomes, significance]
    
  concept:
    required: [name, concept_type]
    optional: [definition, examples, related_concepts]
    
  pattern:
    required: [pattern_type, source_system]
    optional: [parameters, state_vector, generation]
```

### Node YAML Frontmatter

```yaml
---
id: "uuid-v4"
type: "character"
created: "2026-02-21T12:00:00Z"
modified: "2026-02-21T12:30:00Z"
source: "ca_generated|llm_generated|user_created"
embedding_id: "chroma-abc123"
properties:
  name: "Elena Voss"
  role: "Rebel Leader"
  motivation: "Liberate the oppressed"
relations:
  - to: "[[marcus]]"
    type: "ally"
    weight: 0.9
    context: "Fought together at Millhaven"
  - to: "[[millhaven]]"
    type: "origin"
    weight: 0.7
tags: [ca-derived, protagonist, resistance]
---
```

---

## Edge Schema

### Edge Types

```python
EDGE_TYPES = {
    # Ecological
    'predation': {
        'source_types': ['species'],
        'target_types': ['species'],
        'weight_range': (0, 1),
        'bidirectional': False,
        'description': 'Predator-prey relationship'
    },
    'competition': {
        'source_types': ['species', 'faction'],
        'target_types': ['species', 'faction'],
        'weight_range': (0, 1),
        'bidirectional': True,
        'description': 'Resource competition'
    },
    'mutualism': {
        'source_types': ['species'],
        'target_types': ['species'],
        'weight_range': (0, 1),
        'bidirectional': True,
        'description': 'Beneficial relationship'
    },
    
    # Narrative
    'alliance': {
        'source_types': ['character', 'faction'],
        'target_types': ['character', 'faction'],
        'weight_range': (0, 1),
        'bidirectional': True,
        'description': 'Cooperative relationship'
    },
    'conflict': {
        'source_types': ['character', 'faction', 'species'],
        'target_types': ['character', 'faction', 'species'],
        'weight_range': (0, 1),
        'bidirectional': False,
        'description': 'Antagonistic relationship'
    },
    'dependency': {
        'source_types': ['character', 'faction', 'species'],
        'target_types': ['location', 'resource', 'concept'],
        'weight_range': (0, 1),
        'bidirectional': False,
        'description': 'Reliance relationship'
    },
    
    # Spatial/Structural
    'contains': {
        'source_types': ['location', 'faction'],
        'target_types': ['character', 'species', 'location'],
        'weight_range': None,
        'bidirectional': False,
        'description': 'Containment relationship'
    },
    'adjacent': {
        'source_types': ['location'],
        'target_types': ['location'],
        'weight_range': None,
        'bidirectional': True,
        'description': 'Physical proximity'
    },
    
    # Semantic
    'references': {
        'source_types': ['event', 'concept', 'character'],
        'target_types': ['character', 'location', 'event', 'concept'],
        'weight_range': (0, 1),
        'bidirectional': False,
        'description': 'Mentions or references'
    },
    'similar_to': {
        'source_types': ['*'],
        'target_types': ['*'],
        'weight_range': (0, 1),
        'bidirectional': True,
        'description': 'Semantic similarity'
    },
    'causes': {
        'source_types': ['event', 'pattern'],
        'target_types': ['event', 'character', 'species'],
        'weight_range': (0, 1),
        'bidirectional': False,
        'description': 'Causal relationship'
    },
}
```

---

## Processing Modules

### Module Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ModuleResult:
    created_nodes: List[str]
    modified_nodes: List[str]
    created_edges: List[tuple]
    modified_edges: List[tuple]
    metadata: Dict[str, Any]

class KnowledgeModule(ABC):
    """Base class for graph processing modules (BrainSimII-inspired)."""
    
    module_type: str
    input_node_types: List[str]
    output_node_types: List[str]
    
    @abstractmethod
    def process(
        self, 
        graph: 'KnowledgeGraph', 
        node_ids: List[str]
    ) -> ModuleResult:
        """Process nodes and return changes."""
        pass
    
    @abstractmethod
    def get_context(self, graph: 'KnowledgeGraph', node_id: str) -> Dict:
        """Get relevant context for node processing."""
        pass
    
    def validate_inputs(self, graph: 'KnowledgeGraph', node_ids: List[str]) -> bool:
        """Validate that input nodes match expected types."""
        for nid in node_ids:
            node = graph.get_node(nid)
            if node and node.type not in self.input_node_types:
                return False
        return True
```

### Species Evolution Module

```python
class SpeciesEvolutionModule(KnowledgeModule):
    """Process species evolution based on ecological relationships."""
    
    module_type = "species_evolution"
    input_node_types = ["species", "location"]
    output_node_types = ["species", "event"]
    
    def process(self, graph, node_ids) -> ModuleResult:
        # 1. Gather species and their relationships
        # 2. Apply population dynamics
        # 3. Check for speciation events
        # 4. Check for extinction events
        # 5. Update traits via mutation
        # 6. Create events for significant changes
        pass
    
    def get_context(self, graph, node_id) -> Dict:
        node = graph.get_node(node_id)
        if node.type != "species":
            return {}
        
        # Get ecological context
        predators = graph.query_neighbors(
            node_id, 
            edge_types=["predation"],
            direction="in"
        )
        prey = graph.query_neighbors(
            node_id,
            edge_types=["predation"],
            direction="out"
        )
        competitors = graph.query_neighbors(
            node_id,
            edge_types=["competition"]
        )
        location = graph.query_neighbors(
            node_id,
            edge_types=["contains"],
            direction="in"
        )
        
        return {
            "species": node,
            "predators": predators,
            "prey": prey,
            "competitors": competitors,
            "location": location[0] if location else None
        }
```

### Narrative Generation Module

```python
class NarrativeGenerationModule(KnowledgeModule):
    """Generate narrative content from entity relationships."""
    
    module_type = "narrative_generation"
    input_node_types = ["character", "location", "faction", "event"]
    output_node_types = ["event", "concept"]
    
    def process(self, graph, node_ids) -> ModuleResult:
        # 1. Identify conflict clusters
        # 2. Detect story arc patterns
        # 3. Generate events from tensions
        # 4. Create narrative concepts
        pass
    
    def get_context(self, graph, node_id) -> Dict:
        node = graph.get_node(node_id)
        
        # Get narrative context
        allies = graph.query_neighbors(node_id, edge_types=["alliance"])
        enemies = graph.query_neighbors(node_id, edge_types=["conflict"])
        dependencies = graph.query_neighbors(node_id, edge_types=["dependency"])
        
        return {
            "entity": node,
            "allies": allies,
            "enemies": enemies,
            "dependencies": dependencies
        }
```

---

## Storage Layer

### SQLite Schema

```sql
-- Nodes table
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source TEXT DEFAULT 'user_created',
    embedding_id TEXT,
    properties JSON,
    tags JSON
);

-- Edges table
CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL,
    context TEXT,
    bidirectional BOOLEAN DEFAULT 0,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id),
    UNIQUE(source_id, target_id, edge_type)
);

-- Indexes
CREATE INDEX idx_nodes_type ON nodes(type);
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_type ON edges(edge_type);

-- Full-text search
CREATE VIRTUAL TABLE nodes_fts USING fts5(
    id, type, properties, tags
);
```

### ChromaDB Integration

```python
class ChromaBridge:
    """Bridge between KnowledgeGraph and ChromaDB."""
    
    def __init__(self, collection_name: str = "knowledge_nodes"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def index_node(self, node: KnowledgeNode) -> str:
        """Add node to vector index."""
        text = self._node_to_text(node)
        embedding = self.embedder.encode(text)
        
        self.collection.add(
            ids=[node.id],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "type": node.type,
                "source": node.source,
                "created": node.created
            }],
            documents=[text]
        )
        return node.id
    
    def search_similar(
        self, 
        query: str, 
        node_types: List[str] = None,
        k: int = 10
    ) -> List[str]:
        """Find similar nodes by semantic search."""
        embedding = self.embedder.encode(query)
        
        where = {"type": {"$in": node_types}} if node_types else None
        
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
            where=where
        )
        
        return results['ids'][0]
```

### Obsidian Sync

```python
class ObsidianSync:
    """Bidirectional sync with Obsidian vault."""
    
    def __init__(self, vault_path: str, graph: KnowledgeGraph):
        self.vault_path = Path(vault_path)
        self.graph = graph
    
    def export_node(self, node_id: str) -> Path:
        """Export node to Obsidian MD file."""
        node = self.graph.get_node(node_id)
        
        # Determine file path
        type_dir = self.vault_path / "entities" / f"{node.type}s"
        type_dir.mkdir(parents=True, exist_ok=True)
        file_path = type_dir / f"{self._safe_filename(node.properties.get('name', node_id))}.md"
        
        # Build frontmatter
        frontmatter = self._build_frontmatter(node)
        
        # Build content
        content = self._build_content(node)
        
        # Write file
        file_path.write_text(f"---\n{frontmatter}\n---\n\n{content}")
        
        return file_path
    
    def import_vault(self) -> List[str]:
        """Import all MD files from vault to graph."""
        imported = []
        
        for md_file in self.vault_path.rglob("*.md"):
            if md_file.name.startswith("_"):
                continue
            
            try:
                node_id = self.import_file(md_file)
                imported.append(node_id)
            except Exception as e:
                print(f"Failed to import {md_file}: {e}")
        
        return imported
```

---

## API Interface

```python
class KnowledgeGraph:
    """Main knowledge graph interface."""
    
    def __init__(
        self,
        db_path: str = "knowledge.db",
        chroma_path: str = "./chroma_db",
        obsidian_vault: str = None
    ):
        self.db = SQLiteBackend(db_path)
        self.chroma = ChromaBridge(chroma_path)
        self.obsidian = ObsidianSync(obsidian_vault, self) if obsidian_vault else None
        self.modules: Dict[str, KnowledgeModule] = {}
    
    # Node operations
    def add_node(
        self,
        node_type: str,
        properties: Dict,
        relations: List[Dict] = None,
        tags: List[str] = None,
        source: str = "user_created"
    ) -> str:
        """Create a new node and return its ID."""
        pass
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve a node by ID."""
        pass
    
    def update_node(self, node_id: str, properties: Dict) -> bool:
        """Update node properties."""
        pass
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        pass
    
    # Edge operations
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = None,
        context: str = None
    ) -> int:
        """Create an edge between nodes."""
        pass
    
    def query_neighbors(
        self,
        node_id: str,
        edge_types: List[str] = None,
        direction: str = "both",
        depth: int = 1
    ) -> List[KnowledgeNode]:
        """Query neighboring nodes."""
        pass
    
    # Search operations
    def vector_search(
        self,
        query: str,
        node_types: List[str] = None,
        k: int = 10
    ) -> List[KnowledgeNode]:
        """Semantic search across nodes."""
        pass
    
    def text_search(
        self,
        query: str,
        node_types: List[str] = None,
        k: int = 10
    ) -> List[KnowledgeNode]:
        """Full-text search across nodes."""
        pass
    
    # Context building
    def get_context_for_llm(
        self,
        target_node: str,
        max_tokens: int = 2000,
        include_relations: int = 2
    ) -> str:
        """Build LLM-ready context string."""
        pass
    
    # Module operations
    def register_module(self, module: KnowledgeModule):
        """Register a processing module."""
        self.modules[module.module_type] = module
    
    def run_module(
        self,
        module_name: str,
        node_ids: List[str]
    ) -> ModuleResult:
        """Execute a processing module."""
        pass
    
    # Import/Export
    def export_obsidian(self, output_dir: str):
        """Export entire graph to Obsidian vault."""
        pass
    
    def import_obsidian(self, vault_dir: str):
        """Import Obsidian vault to graph."""
        pass
    
    def to_json(self) -> Dict:
        """Export graph to JSON."""
        pass
    
    @classmethod
    def from_json(cls, data: Dict) -> 'KnowledgeGraph':
        """Import graph from JSON."""
        pass
```

---

## File Structure

```
AlchemicalLab/
├── GraphEngine/                    # Knowledge Graph System
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py      # Main graph class
│   │   ├── node_schema.py          # Node definitions
│   │   ├── edge_schema.py          # Edge definitions
│   │   └── types.py                # Type definitions
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── sqlite_backend.py       # SQLite storage
│   │   ├── chroma_bridge.py        # Vector DB bridge
│   │   └── obsidian_sync.py        # MD file sync
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── base_module.py          # Abstract base
│   │   ├── species_evolution.py    # Evolution module
│   │   └── narrative_generation.py # Narrative module
│   ├── bridges/
│   │   ├── __init__.py
│   │   ├── calab_bridge.py         # CALab integration
│   │   └── storylab_bridge.py      # StoryLab integration
│   └── tests/
│       ├── __init__.py
│       ├── test_knowledge_graph.py
│       ├── test_storage.py
│       └── test_modules.py
│
├── KnowledgeVault/                 # Obsidian Vault
│   ├── _templates/
│   │   ├── character.md
│   │   ├── species.md
│   │   ├── location.md
│   │   ├── faction.md
│   │   └── event.md
│   ├── entities/
│   │   ├── characters/
│   │   ├── species/
│   │   ├── locations/
│   │   └── factions/
│   ├── concepts/
│   │   ├── themes/
│   │   └── patterns/
│   ├── events/
│   │   └── timeline/
│   └── .obsidian/
│       ├── app.json
│       └── graph.json
```

---

## Implementation Phases

### Phase 1: Core (Estimated: 2-3 days)

1. Create directory structure
2. Implement node_schema.py and edge_schema.py
3. Implement sqlite_backend.py
4. Implement chroma_bridge.py
5. Implement knowledge_graph.py core methods
6. Write tests

### Phase 2: Modules (Estimated: 2-3 days)

1. Implement base_module.py
2. Implement species_evolution.py
3. Implement narrative_generation.py
4. Create module pipeline
5. Write tests

### Phase 3: Bridges (Estimated: 1-2 days)

1. Implement calab_bridge.py
2. Implement storylab_bridge.py
3. Implement obsidian_sync.py
4. Integrate with existing systems
5. Write tests

### Phase 4: Visualization (Estimated: 2-3 days)

1. Create web server with Flask/FastAPI
2. Implement D3.js frontend
3. Add WebSocket for real-time updates
4. Configure Obsidian graph view

---

## Success Criteria

- [ ] Nodes can be created, updated, deleted via API
- [ ] Edges connect nodes with typed relationships
- [ ] Vector search returns semantically similar nodes
- [ ] Obsidian sync exports/imports correctly
- [ ] Species evolution module processes ecosystem nodes
- [ ] Narrative generation module creates story events
- [ ] CALab bridge converts WorldState to graph
- [ ] StoryLab bridge generates LLM context from graph
- [ ] All tests pass
- [ ] Documentation complete

---

## References

- BrainSimII: https://github.com/FutureAIGuru/BrainSimII
- Obsidian: https://obsidian.md
- ChromaDB: https://www.trychroma.com
- NetworkX: https://networkx.org

---

*Plan created by: opencode/glm-5-free*  
*Date: 2026-02-21*
