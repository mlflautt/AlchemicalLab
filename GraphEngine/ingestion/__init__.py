"""
GraphEngine Ingestion Module.

Text-to-world-graph ingestion pipeline for novels, scripts, and prompts.
"""

from GraphEngine.ingestion.chunker import (
    TextChunker,
    TextChunk,
    IngestOptions,
    load_text_file,
    detect_source_type,
)

from GraphEngine.ingestion.deterministic.regex_extractor import (
    DeterministicExtractor,
    extract_quick_entities,
)

from GraphEngine.ingestion.resolver import EntityResolver, resolve_entities_simple

from GraphEngine.ingestion.validator import ValidatorLLM, validate_entities_simple

from GraphEngine.ingestion.llm_extractors.base_extractor import (
    LLMExtractorPipeline,
    create_extractor_pipeline,
)

from GraphEngine.ingestion.ingestor import (
    WorldGraphIngestor,
    IngestResult,
    ingest_text,
    ingest_file,
)

__all__ = [
    'TextChunker',
    'TextChunk',
    'IngestOptions',
    'load_text_file',
    'detect_source_type',
    'DeterministicExtractor',
    'extract_quick_entities',
    'EntityResolver',
    'resolve_entities_simple',
    'ValidatorLLM',
    'validate_entities_simple',
    'LLMExtractorPipeline',
    'create_extractor_pipeline',
    'WorldGraphIngestor',
    'IngestResult',
    'ingest_text',
    'ingest_file',
]
