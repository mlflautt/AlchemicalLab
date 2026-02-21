"""
Tests for GraphEngine Knowledge Graph.
"""

import pytest
import tempfile
import os
from pathlib import Path

from GraphEngine import (
    KnowledgeGraph, KnowledgeNode, KnowledgeEdge,
    NodeType, EdgeType, NodeSchema, EdgeSchema
)
from GraphEngine.core.types import RelationDef
from GraphEngine.modules import SpeciesEvolutionModule, NarrativeGenerationModule
from GraphEngine.bridges import CALabBridge, StoryLabBridge


class TestKnowledgeGraph:
    """Test cases for KnowledgeGraph class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def graph(self, temp_db):
        """Create a KnowledgeGraph instance."""
        return KnowledgeGraph(db_path=temp_db)
    
    def test_add_node(self, graph):
        """Test adding a node."""
        node_id = graph.add_node(
            node_type='character',
            properties={'name': 'Elena', 'role': 'protagonist'},
            tags=['hero']
        )
        
        assert node_id is not None
        
        node = graph.get_node(node_id)
        assert node is not None
        assert node.type == 'character'
        assert node.properties['name'] == 'Elena'
        assert 'hero' in node.tags
    
    def test_update_node(self, graph):
        """Test updating a node."""
        node_id = graph.add_node(
            node_type='species',
            properties={'name': 'Dragon', 'species_type': 'predator', 'population': 100}
        )
        
        graph.update_node(node_id, {'population': 150, 'fitness': 0.8})
        
        node = graph.get_node(node_id)
        assert node.properties['population'] == 150
        assert node.properties['fitness'] == 0.8
    
    def test_delete_node(self, graph):
        """Test deleting a node."""
        node_id = graph.add_node(
            node_type='location',
            properties={'name': 'Forest', 'location_type': 'wilderness'}
        )
        
        assert graph.get_node(node_id) is not None
        
        result = graph.delete_node(node_id)
        assert result is True
        assert graph.get_node(node_id) is None
    
    def test_add_edge(self, graph):
        """Test adding an edge."""
        char_id = graph.add_node(
            node_type='character',
            properties={'name': 'Hero', 'role': 'protagonist'}
        )
        loc_id = graph.add_node(
            node_type='location',
            properties={'name': 'Castle', 'location_type': 'building'}
        )
        
        edge_id = graph.add_edge(
            source_id=char_id,
            target_id=loc_id,
            edge_type='origin',
            weight=0.9,
            context='Born here'
        )
        
        assert edge_id is not None
        
        edges = graph.get_edges(char_id)
        assert len(edges) == 1
        assert edges[0].edge_type == 'origin'
    
    def test_search(self, graph):
        """Test searching for nodes."""
        graph.add_node(
            node_type='character',
            properties={'name': 'Alice', 'role': 'wizard'},
            tags=['magic-user']
        )
        graph.add_node(
            node_type='character',
            properties={'name': 'Bob', 'role': 'warrior'},
            tags=['fighter']
        )
        
        results = graph.list_nodes(node_type='character')
        assert len(results) == 2
    
    def test_get_neighbors(self, graph):
        """Test getting neighbors."""
        a_id = graph.add_node(node_type='character', properties={'name': 'A', 'role': 'x'})
        b_id = graph.add_node(node_type='character', properties={'name': 'B', 'role': 'x'})
        c_id = graph.add_node(node_type='character', properties={'name': 'C', 'role': 'x'})
        
        graph.add_edge(a_id, b_id, 'alliance', weight=0.8)
        graph.add_edge(b_id, c_id, 'alliance', weight=0.6)
        
        neighbors = graph.get_neighbors(a_id, depth=1)
        assert b_id in neighbors
        assert c_id not in neighbors
        
        neighbors = graph.get_neighbors(a_id, depth=2)
        assert b_id in neighbors
        assert c_id in neighbors
    
    def test_find_path(self, graph):
        """Test finding path between nodes."""
        a_id = graph.add_node(node_type='character', properties={'name': 'A', 'role': 'x'})
        b_id = graph.add_node(node_type='character', properties={'name': 'B', 'role': 'x'})
        c_id = graph.add_node(node_type='character', properties={'name': 'C', 'role': 'x'})
        
        graph.add_edge(a_id, b_id, 'alliance')
        graph.add_edge(b_id, c_id, 'alliance')
        
        path = graph.find_path(a_id, c_id)
        assert path == [a_id, b_id, c_id]


class TestSpeciesEvolutionModule:
    """Test cases for SpeciesEvolutionModule."""
    
    @pytest.fixture
    def graph(self):
        """Create a KnowledgeGraph instance."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        graph = KnowledgeGraph(db_path=db_path)
        yield graph
        os.unlink(db_path)
    
    def test_process_ecosystem(self, graph):
        """Test processing ecosystem data."""
        module = SpeciesEvolutionModule()
        
        ecosystem_state = {
            'species': [
                {
                    'id': 'species_1',
                    'name': 'Wolf',
                    'type': 'predator',
                    'population': 100,
                    'fitness': 0.7,
                    'traits': {'strength': 0.8, 'speed': 0.6}
                },
                {
                    'id': 'species_2',
                    'name': 'Deer',
                    'type': 'prey',
                    'population': 500,
                    'fitness': 0.5,
                    'traits': {'speed': 0.9, 'stealth': 0.4}
                }
            ],
            'relationships': [
                {'source': 'species_1', 'target': 'species_2', 'type': 'predation', 'strength': 0.8}
            ]
        }
        
        result = module.process(graph, ecosystem_state=ecosystem_state, generation=1)
        
        assert len(result.created_nodes) == 2
        assert len(result.created_edges) == 1
        
        wolf = graph.get_node('species_1')
        assert wolf is not None
        assert wolf.properties['name'] == 'Wolf'
        assert wolf.type == 'species'


class TestNarrativeGenerationModule:
    """Test cases for NarrativeGenerationModule."""
    
    @pytest.fixture
    def graph(self):
        """Create a KnowledgeGraph instance."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        graph = KnowledgeGraph(db_path=db_path)
        yield graph
        os.unlink(db_path)
    
    def test_process_characters(self, graph):
        """Test processing character data."""
        module = NarrativeGenerationModule()
        
        narrative_state = {
            'characters': [
                {
                    'id': 'char_1',
                    'name': 'Elena',
                    'role': 'protagonist',
                    'motivation': 'Save the kingdom',
                    'is_protagonist': True
                },
                {
                    'id': 'char_2',
                    'name': 'Dark Lord',
                    'role': 'antagonist',
                    'motivation': 'Conquer the world',
                    'is_antagonist': True
                }
            ]
        }
        
        result = module.process(graph, narrative_state=narrative_state, generation=1)
        
        assert len(result.created_nodes) == 2
        
        elena = graph.get_node('char_1')
        assert elena is not None
        assert 'protagonist' in elena.tags


class TestNodeSchema:
    """Test cases for NodeSchema."""
    
    def test_validate_character(self):
        """Test validating a character node."""
        schema = NodeSchema()
        
        is_valid, errors, normalized = schema.validate(
            'character',
            {'name': 'Hero', 'role': 'protagonist'}
        )
        
        assert is_valid
        assert len(errors) == 0
        assert normalized['name'] == 'Hero'
    
    def test_validate_missing_required(self):
        """Test validation with missing required properties."""
        schema = NodeSchema()
        
        is_valid, errors, normalized = schema.validate(
            'character',
            {'name': 'Hero'}
        )
        
        assert not is_valid
        assert any('role' in e for e in errors)
    
    def test_list_types(self):
        """Test listing node types."""
        schema = NodeSchema()
        types = schema.list_types()
        
        assert 'character' in types
        assert 'species' in types
        assert 'location' in types


class TestEdgeSchema:
    """Test cases for EdgeSchema."""
    
    def test_validate_edge(self):
        """Test validating an edge."""
        schema = EdgeSchema()
        
        is_valid, errors, normalized_weight, bidirectional = schema.validate(
            'alliance',
            'character',
            'character',
            0.8
        )
        
        assert is_valid
        assert bidirectional is True
    
    def test_invalid_edge_type(self):
        """Test validation with invalid edge type."""
        schema = EdgeSchema()
        
        is_valid, errors, _, _ = schema.validate(
            'invalid_type',
            'character',
            'character',
            0.5
        )
        
        assert not is_valid
    
    def test_get_compatible_edges(self):
        """Test getting compatible edge types."""
        schema = EdgeSchema()
        
        compatible = schema.get_compatible_edges('species', 'species')
        
        assert 'predation' in compatible
        assert 'competition' in compatible
        assert 'mutualism' in compatible


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
