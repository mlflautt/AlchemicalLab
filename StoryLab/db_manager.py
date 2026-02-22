#!/usr/bin/env python3
"""
DB Manager: Integrates Chroma vector DB for story embeddings and NetworkX for graph connections.
"""

import chromadb
from sentence_transformers import SentenceTransformer
import networkx as nx
import json
import os

class StoryDBManager:
    def __init__(self, chroma_path="./chroma_db", model_name="all-MiniLM-L6-v2"):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name="stories")
        self.embedder = SentenceTransformer(model_name)
        self.graph = nx.Graph()
        self.load_existing()

    def load_existing(self):
        """Load existing stories into graph."""
        try:
            results = self.collection.get(include=["metadatas", "documents"])
            if results and 'ids' in results:
                ids = results['ids']
                metadatas = results.get('metadatas', [])
                documents = results.get('documents', [])

                for i, story_id in enumerate(ids):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    if metadata is None:
                        metadata = {}
                    doc = documents[i] if i < len(documents) else ''
                    self.graph.add_node(story_id, **metadata, text=doc)
        except Exception as e:
            print(f"Error loading existing stories: {e}")
            # Continue without existing data

    def add_story(self, story_id, story_text, metadata=None):
        """Add story to Chroma and NetworkX."""
        if metadata is None:
            metadata = {}
        embedding = self.embedder.encode(story_text).tolist()
        self.collection.add(
            ids=[story_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[story_text]
        )
        self.graph.add_node(story_id, **metadata, text=story_text, type='story')
        self.build_entity_graph()  # Update entities

    def query_similar(self, query_text, n_results=5):
        """Query similar stories."""
        query_emb = self.embedder.encode(query_text).tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        return results

    def extract_entities(self, story_text):
        """LLM-based entity extraction."""
        import subprocess
        prompt = f"Extract key entities (characters, locations, factions) from this story. List as comma-separated.\n\nStory:\n{story_text}\n\nEntities:"
        cmd = ['../llama.cpp/build/bin/llama-simple', '-m', '../qwen_q2.gguf', '-ngl', '22', '-n', '128', prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        entities = result.stdout.strip().split(',')
        return [e.strip() for e in entities if e.strip()]

    def extract_relationships(self, story_text):
        """Extract relationships between entities."""
        import subprocess
        prompt = f"From this story, list relationships between entities as 'entity1 - relation - entity2'. One per line.\n\nStory:\n{story_text}\n\nRelationships:"
        cmd = ['../llama.cpp/build/bin/llama-simple', '-m', '../qwen_q2.gguf', '-ngl', '22', '-n', '256', prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        lines = result.stdout.strip().split('\n')
        relations = []
        for line in lines:
            if ' - ' in line:
                parts = line.split(' - ')
                if len(parts) == 3:
                    relations.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
        return relations

    def build_entity_graph(self):
        """Build graph with entities and relationships."""
        for node in list(self.graph.nodes()):
            if self.graph.nodes[node].get('type') == 'story':
                entities = self.extract_entities(self.graph.nodes[node]['text'])
                for ent in entities:
                    if ent not in self.graph:
                        self.graph.add_node(ent, type='entity')
                    self.graph.add_edge(node, ent, type='contains')
                relations = self.extract_relationships(self.graph.nodes[node]['text'])
                for e1, rel, e2 in relations:
                    if e1 in self.graph and e2 in self.graph:
                        self.graph.add_edge(e1, e2, type=rel)

    def build_connections(self, threshold=0.8):
        """Build edges in graph based on similarity."""
        ids = list(self.graph.nodes())
        embeddings = []
        for id in ids:
            result = self.collection.get(ids=[id], include=["embeddings"])
            if result and 'embeddings' in result and result['embeddings']:
                emb = result['embeddings'][0]
                embeddings.append(emb)
            else:
                embeddings.append([0.0] * 384)  # Default embedding size
        for i, id1 in enumerate(ids):
            for j, id2 in enumerate(ids):
                if i < j:
                    sim = self.cosine_similarity(embeddings[i], embeddings[j])
                    if sim > threshold:
                        self.graph.add_edge(id1, id2, weight=sim)

    def cosine_similarity(self, a, b):
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_graph_data(self):
        """Return graph data for visualization."""
        nodes = [{"id": n, **self.graph.nodes[n]} for n in self.graph.nodes()]
        links = [{"source": u, "target": v, "value": d["weight"]} for u, v, d in self.graph.edges(data=True)]
        return {"nodes": nodes, "links": links}

if __name__ == "__main__":
    db = StoryDBManager()
    # Test
    db.add_story("test1", "A hero saves the world.", {"genre": "fantasy"})
    db.add_story("test2", "A warrior fights dragons.", {"genre": "fantasy"})
    db.build_connections()
    print("Graph:", db.get_graph_data())