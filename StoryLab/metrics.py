#!/usr/bin/env python3
"""
Metrics: Evaluate story quality with graph density, coherence, and comparison to classics.
"""

import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np

class StoryMetrics:
    def __init__(self, embedder_model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedder_model)
        # Embeddings of classic stories
        self.classics = {
            "LOTR": "In a land called Middle-earth, a hobbit named Frodo Baggins inherits a powerful ring from his uncle Bilbo. With the help of a fellowship, he embarks on a quest to destroy the ring in Mount Doom, facing dark forces led by Sauron.",
            "Dune": "On the desert planet Arrakis, young Paul Atreides becomes embroiled in political intrigue after his family takes control of the spice production. He discovers his destiny amidst sandworms and interstellar conflict."
        }
        self.classic_embeddings = {k: self.embedder.encode(v) for k, v in self.classics.items()}

    def graph_density(self, graph):
        """Compute graph density."""
        return nx.density(graph)

    def coherence_score(self, story_text, world_dna):
        """Simple coherence: similarity to world-DNA."""
        story_emb = self.embedder.encode(story_text)
        dna_emb = self.embedder.encode(world_dna)
        return np.dot(story_emb, dna_emb) / (np.linalg.norm(story_emb) * np.linalg.norm(dna_emb))

    def classic_similarity(self, story_text):
        """Similarity to LOTR and Dune."""
        story_emb = self.embedder.encode(story_text)
        similarities = {}
        for name, emb in self.classic_embeddings.items():
            sim = np.dot(story_emb, emb) / (np.linalg.norm(story_emb) * np.linalg.norm(emb))
            similarities[name] = sim
        return similarities

    def evaluate_story(self, story_text, world_dna, graph):
        """Full evaluation."""
        density = float(self.graph_density(graph)) if graph else 0.0
        coherence = float(self.coherence_score(story_text, world_dna))
        similarities = {k: float(v) for k, v in self.classic_similarity(story_text).items()}
        return {
            "graph_density": density,
            "coherence": coherence,
            "classic_similarities": similarities
        }

if __name__ == "__main__":
    metrics = StoryMetrics()
    # Test
    test_story = "A hero saves the world from evil."
    test_dna = "Young engineer discovers artifact."
    test_graph = nx.Graph()
    test_graph.add_nodes_from([1,2,3])
    test_graph.add_edges_from([(1,2), (2,3)])
    result = metrics.evaluate_story(test_story, test_dna, test_graph)
    print("Metrics:", result)