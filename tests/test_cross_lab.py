"""
Cross-Lab Integration Tests
===========================

Tests for the CA → StoryLab integration pipeline.
"""

import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHybridCAAggregator(unittest.TestCase):
    """Tests for the hybrid CA aggregator."""
    
    def test_import(self):
        """Test that hybrid_ca_aggregator can be imported."""
        from CALab.hybrid_ca_aggregator import HybridCAAggregator, WorldState
        self.assertIsNotNone(HybridCAAggregator)
        self.assertIsNotNone(WorldState)
    
    def test_world_state_creation(self):
        """Test WorldState dataclass creation."""
        from CALab.hybrid_ca_aggregator import WorldState
        
        ws = WorldState(
            generation=10,
            world_size=(50, 50)
        )
        
        self.assertEqual(ws.generation, 10)
        self.assertEqual(ws.world_size, (50, 50))
        self.assertEqual(len(ws.species), 0)
        self.assertEqual(len(ws.characters), 0)
    
    def test_world_state_serialization(self):
        """Test WorldState serialization to dict/json."""
        from CALab.hybrid_ca_aggregator import WorldState, SpeciesData
        
        ws = WorldState(
            generation=5,
            world_size=(30, 30)
        )
        ws.species[1] = SpeciesData(
            species_id=1,
            name="Test Species",
            species_type="producer",
            center=(10, 10),
            population=100,
            traits={"speed": 0.5},
            preferred_biomes=["forest"],
            fitness=0.8,
            generation_alive=5
        )
        
        d = ws.to_dict()
        self.assertIn('generation', d)
        self.assertIn('species', d)
        self.assertEqual(d['generation'], 5)
        self.assertIn('1', d['species'])
        
        j = ws.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed['generation'], 5)


class TestNarrativeBridge(unittest.TestCase):
    """Tests for the narrative bridge."""
    
    def test_import(self):
        """Test that narrative_bridge can be imported."""
        from CALab.narrative_bridge import WorldDNAGenerator, world_state_to_world_dna
        self.assertIsNotNone(WorldDNAGenerator)
        self.assertIsNotNone(world_state_to_world_dna)
    
    def test_world_dna_generation(self):
        """Test generation of world_dna from WorldState."""
        from CALab.narrative_bridge import WorldDNAGenerator
        from CALab.hybrid_ca_aggregator import WorldState, SpeciesData, CharacterData
        
        generator = WorldDNAGenerator()
        
        ws = WorldState(
            generation=20,
            world_size=(40, 40)
        )
        ws.species[1] = SpeciesData(
            species_id=1,
            name="Test Animal",
            species_type="herbivore",
            center=(15, 15),
            population=50,
            traits={"speed": 0.7},
            preferred_biomes=["grassland"],
            fitness=0.9,
            generation_alive=10
        )
        ws.characters[1] = CharacterData(
            element_id=1,
            name="Test Hero",
            element_type="character",
            center=(20, 20),
            properties={"class": "Warrior", "motivation": "Honor"},
            backstory=["A brave warrior seeking glory."],
            relationships=set(),
            narrative_weight=1.0
        )
        
        dna = generator.generate_world_dna(ws, world_name="Test World")
        
        self.assertIn("# World-DNA", dna)
        self.assertIn("Test World", dna)
        self.assertIn("## Metadata", dna)
        self.assertIn("## Species Parameters", dna)


class TestCAAdapter(unittest.TestCase):
    """Tests for the StoryLab CA adapter."""
    
    def test_import(self):
        """Test that ca_adapter can be imported."""
        from StoryLab.ca_adapter import CAAdapter, CAStoryContext
        self.assertIsNotNone(CAAdapter)
        self.assertIsNotNone(CAStoryContext)
    
    @patch('StoryLab.ca_adapter.CAAdapter._init_db')
    def test_adapter_initialization(self, mock_init_db):
        """Test adapter initialization."""
        from StoryLab.ca_adapter import CAAdapter
        
        adapter = CAAdapter(auto_initialize_db=False)
        self.assertIsNotNone(adapter)
        self.assertIsNone(adapter.db)


class TestUnifiedAPIExtensions(unittest.TestCase):
    """Tests for the extended UnifiedGenerativeAPI."""
    
    def test_import(self):
        """Test that generative_api can be imported."""
        from generative_api import get_generative_api, UnifiedGenerativeAPI
        self.assertIsNotNone(get_generative_api)
        self.assertIsNotNone(UnifiedGenerativeAPI)
    
    def test_api_instance(self):
        """Test that API instance is created correctly."""
        from generative_api import get_generative_api
        
        api = get_generative_api()
        self.assertIsNotNone(api)
        self.assertIn('story', api.get_available_pipelines())
    
    def test_generate_world_from_ca_method_exists(self):
        """Test that generate_world_from_ca method exists."""
        from generative_api import UnifiedGenerativeAPI
        
        api = UnifiedGenerativeAPI()
        self.assertTrue(hasattr(api, 'generate_world_from_ca'))
        self.assertTrue(callable(api.generate_world_from_ca))
    
    def test_simulate_and_narrate_method_exists(self):
        """Test that simulate_and_narrate method exists."""
        from generative_api import UnifiedGenerativeAPI
        
        api = UnifiedGenerativeAPI()
        self.assertTrue(hasattr(api, 'simulate_and_narrate'))
        self.assertTrue(callable(api.simulate_and_narrate))


class TestIntegrationFlow(unittest.TestCase):
    """Integration tests for the complete CA → Story flow."""
    
    def test_dataclass_imports(self):
        """Test all dataclass imports work together."""
        from CALab.hybrid_ca_aggregator import (
            WorldState, SpeciesData, CharacterData, LocationData,
            FactionData, StoryArcData, EcologicalRelationshipData,
            NarrativeRelationshipData
        )
        
        ws = WorldState(generation=1, world_size=(10, 10))
        self.assertIsNotNone(ws)
        
        species = SpeciesData(
            species_id=1, name="Test", species_type="producer",
            center=(5, 5), population=100, traits={},
            preferred_biomes=[], fitness=0.5, generation_alive=1
        )
        self.assertEqual(species.name, "Test")
    
    def test_mini_simulation(self):
        """Test a minimal simulation run."""
        try:
            from CALab.hybrid_ca_aggregator import HybridCAAggregator
            
            aggregator = HybridCAAggregator(world_size=(30, 30))
            aggregator.initialize(density=0.3, seed=42)
            
            for _ in range(5):
                aggregator.step()
            
            ws = aggregator.get_world_state()
            self.assertGreaterEqual(ws.generation, 5)
            
        except ImportError as e:
            self.skipTest(f"Dependencies not available: {e}")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestHybridCAAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestNarrativeBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestCAAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedAPIExtensions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationFlow))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cross-lab integration tests")
    parser.add_argument('--quick', action='store_true', help='Run only quick tests')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cross-Lab Integration Tests")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
