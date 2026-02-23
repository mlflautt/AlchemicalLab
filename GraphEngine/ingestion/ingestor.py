"""
World Graph Ingestor - Main ingestion pipeline.

Orchestrates the complete text-to-knowledge-graph pipeline.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import os

from GraphEngine.ingestion.chunker import TextChunker, TextChunk, IngestOptions, load_text_file, detect_source_type
from GraphEngine.ingestion.deterministic.regex_extractor import DeterministicExtractor, extract_quick_entities
from GraphEngine.ingestion.llm_extractors.base_extractor import LLMExtractorPipeline, create_extractor_pipeline
from GraphEngine.ingestion.resolver import EntityResolver, resolve_entities_simple
from GraphEngine.ingestion.validator import ValidatorLLM, validate_entities_simple


@dataclass
class IngestResult:
    """Result of an ingestion operation."""
    success: bool
    source: str
    source_type: str
    chunks_processed: int
    entities_created: Dict[str, int]
    relationships_created: int
    narrative_arcs: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'source': self.source,
            'source_type': self.source_type,
            'chunks_processed': self.chunks_processed,
            'entities_created': self.entities_created,
            'relationships_created': self.relationships_created,
            'narrative_arcs': self.narrative_arcs,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
        }


class WorldGraphIngestor:
    """
    Main ingestion pipeline for converting text to knowledge graph.
    
    Pipeline:
    1. Load text (file or string)
    2. Chunk text into sections
    3. Extract deterministic entities (fast)
    4. Extract LLM entities (detailed)
    5. Resolve duplicates
    6. Validate entities
    7. Analyze narrative
    8. Store in knowledge graph
    """
    
    def __init__(
        self,
        graph: 'KnowledgeGraph' = None,
        llm_client: Callable = None,
        options: IngestOptions = None
    ):
        self.graph = graph
        self.llm_client = llm_client
        self.options = options or IngestOptions()
        
        self.chunker = TextChunker()
        self.deterministic = DeterministicExtractor()
        self.llm_extractor = create_extractor_pipeline(llm_client)
        self.resolver = EntityResolver()
        self.validator = ValidatorLLM(llm_client)
    
    def ingest(
        self,
        source: str,
        source_type: str = "novel",
        options: IngestOptions = None
    ) -> IngestResult:
        """
        Main ingestion entry point.
        
        Args:
            source: File path or text string
            source_type: Type of source ("novel", "script", "prompt")
            options: Optional ingestion options
        
        Returns:
            IngestResult with statistics
        """
        opts = options or self.options
        errors = []
        warnings = []
        
        try:
            if os.path.exists(source):
                text, metadata = load_text_file(source)
                source_type = detect_source_type(text, source)
            else:
                text = source
                metadata = {'word_count': len(text.split())}
            
            chunks = self.chunker.chunk(
                text,
                source_type=source_type,
                chunk_by=opts.chunk_by,
                max_tokens=opts.max_chunk_tokens,
                overlap=opts.chunk_overlap
            )
            
            all_entities = {
                'characters': [],
                'locations': [],
                'factions': [],
                'items': [],
                'events': [],
                'relationships': [],
            }
            
            for chunk in chunks:
                chunk_entities = self._process_chunk(chunk, opts)
                
                for key in all_entities:
                    all_entities[key].extend(chunk_entities.get(key, []))
            
            resolved = self.resolver.resolve(all_entities.get('characters', []) + 
                                           all_entities.get('locations', []) +
                                           all_entities.get('factions', []))
            
            resolved_entities = {
                'characters': [e for e in resolved['entities'] if e.get('type') == 'character'],
                'locations': [e for e in resolved['entities'] if e.get('type') == 'location'],
                'factions': [e for e in resolved['entities'] if e.get('type') == 'faction'],
            }
            
            if opts.validate and self.llm_client:
                validation = self.validator.validate(resolved_entities, text)
                if not validation.is_valid:
                    warnings.extend([c['description'] for c in validation.conflicts[:5]])
            
            if self.graph:
                entities_created = self._store_in_graph(resolved_entities, all_entities.get('relationships', []))
            else:
                entities_created = {
                    'characters': len(resolved_entities.get('characters', [])),
                    'locations': len(resolved_entities.get('locations', [])),
                    'factions': len(resolved_entities.get('factions', [])),
                    'items': len(resolved_entities.get('items', [])),
                    'events': len(resolved_entities.get('events', [])),
                }
            
            return IngestResult(
                success=True,
                source=source,
                source_type=source_type,
                chunks_processed=len(chunks),
                entities_created=entities_created,
                relationships_created=len(all_entities.get('relationships', [])),
                narrative_arcs=0,
                warnings=warnings,
                metadata={
                    'word_count': metadata.get('word_count', 0),
                    'chunks': len(chunks),
                    'duplicates_resolved': resolved.get('duplicates_resolved', 0),
                }
            )
        
        except Exception as e:
            return IngestResult(
                success=False,
                source=source,
                source_type=source_type,
                chunks_processed=0,
                entities_created={},
                relationships_created=0,
                narrative_arcs=0,
                errors=[str(e)]
            )
    
    def _process_chunk(
        self,
        chunk: TextChunk,
        options: IngestOptions
    ) -> Dict[str, List]:
        """Process a single text chunk."""
        entities = {
            'characters': [],
            'locations': [],
            'factions': [],
            'items': [],
            'events': [],
            'relationships': [],
        }
        
        if options.extract_deterministic:
            det_result = self.deterministic.extract(chunk.content, chunk.metadata)
            
            for name_data in det_result.get('named_entities', [])[:10]:
                entities['characters'].append({
                    'name': name_data['name'],
                    'type': 'character',
                    'properties': {'source': 'deterministic'},
                    'chunk_id': chunk.chunk_id,
                })
            
            for loc in det_result.get('location_references', [])[:5]:
                entities['locations'].append({
                    'name': loc.get('location', ''),
                    'type': 'location',
                    'properties': {'context': loc.get('context', '')},
                    'chunk_id': chunk.chunk_id,
                })
        
        if options.extract_with_llm and self.llm_client:
            llm_result = self.llm_extractor.extract_all(chunk.content, chunk.chunk_id)
            
            entities['characters'].extend([
                {**c, 'chunk_id': chunk.chunk_id} 
                for c in llm_result.characters
            ])
            entities['locations'].extend([
                {**l, 'chunk_id': chunk.chunk_id}
                for l in llm_result.locations
            ])
            entities['factions'].extend([
                {**f, 'chunk_id': chunk.chunk_id}
                for f in llm_result.factions
            ])
            entities['relationships'].extend(llm_result.relationships)
        
        return entities
    
    def _store_in_graph(
        self,
        entities: Dict[str, List],
        relationships: List[Dict]
    ) -> Dict[str, int]:
        """Store entities in knowledge graph."""
        if not self.graph:
            return {}
        
        counts = {}
        
        for char in entities.get('characters', []):
            char_id = self._generate_node_id(char.get('name', ''), 'character')
            self.graph.add_node(
                node_type='character',
                properties={
                    'name': char.get('name', ''),
                    'role': char.get('role', 'unknown'),
                    'description': char.get('description', ''),
                    'traits': char.get('traits', []),
                    'motivation': char.get('motivation', ''),
                },
                tags=['ingested', 'character'],
                source='text_ingestion',
                node_id=char_id
            )
            counts['characters'] = counts.get('characters', 0) + 1
        
        for loc in entities.get('locations', []):
            loc_id = self._generate_node_id(loc.get('name', ''), 'location')
            self.graph.add_node(
                node_type='location',
                properties={
                    'name': loc.get('name', ''),
                    'location_type': loc.get('type', 'unknown'),
                    'description': loc.get('description', ''),
                    'region': loc.get('region', ''),
                },
                tags=['ingested', 'location'],
                source='text_ingestion',
                node_id=loc_id
            )
            counts['locations'] = counts.get('locations', 0) + 1
        
        for faction in entities.get('factions', []):
            fac_id = self._generate_node_id(faction.get('name', ''), 'faction')
            self.graph.add_node(
                node_type='faction',
                properties={
                    'name': faction.get('name', ''),
                    'ideology': faction.get('ideology', ''),
                    'goals': faction.get('goals', ''),
                },
                tags=['ingested', 'faction'],
                source='text_ingestion',
                node_id=fac_id
            )
            counts['factions'] = counts.get('factions', 0) + 1
        
        for rel in relationships:
            try:
                source_id = self._generate_node_id(rel.get('source', ''), 'unknown')
                target_id = self._generate_node_id(rel.get('target', ''), 'unknown')
                
                if self.graph.get_node(source_id) and self.graph.get_node(target_id):
                    self.graph.add_edge(
                        source_id=source_id,
                        target_id=target_id,
                        edge_type=rel.get('relationship_type', 'references'),
                        weight=rel.get('strength', 0.5),
                        context=rel.get('context', '')
                    )
            except:
                pass
        
        return counts
    
    def _generate_node_id(self, name: str, prefix: str) -> str:
        """Generate a deterministic node ID."""
        clean = ''.join(c.lower() for c in name if c.isalnum() or c.isspace())
        clean = '_'.join(clean.split())
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:6]
        return f"{prefix}_{clean}_{hash_suffix}"


def ingest_text(
    text: str,
    graph: 'KnowledgeGraph' = None,
    llm_client: Callable = None,
    source_type: str = "novel"
) -> IngestResult:
    """
    Convenience function to ingest text directly.
    
    Args:
        text: Text to ingest
        graph: KnowledgeGraph to populate
        llm_client: Optional LLM client
        source_type: Type of source
    
    Returns:
        IngestResult
    """
    ingestor = WorldGraphIngestor(graph, llm_client)
    return ingestor.ingest(text, source_type)


def ingest_file(
    filepath: str,
    graph: 'KnowledgeGraph' = None,
    llm_client: Callable = None
) -> IngestResult:
    """
    Convenience function to ingest a file.
    
    Args:
        filepath: Path to text file
        graph: KnowledgeGraph to populate
        llm_client: Optional LLM client
    
    Returns:
        IngestResult
    """
    ingestor = WorldGraphIngestor(graph, llm_client)
    return ingestor.ingest(filepath)
