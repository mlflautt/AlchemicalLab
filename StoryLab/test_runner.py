#!/usr/bin/env python3
"""
Test Runner: Automates generation testing for quality and performance evaluation.
"""

import time
import json
import os
import psutil
import sys
sys.path.append(os.path.dirname(__file__))
from story_generator import load_world_dna, generate_story_idea
from db_manager import StoryDBManager
from metrics import StoryMetrics

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def run_tests(num_tests=10, multi_step_ratio=0.5):
    """Run automated tests."""
    db = StoryDBManager()
    metrics_calc = StoryMetrics()
    dna = load_world_dna("world_dna.md")
    results = []
    
    for i in range(num_tests):
        multi_step = (i % int(1/multi_step_ratio)) == 0 if multi_step_ratio > 0 else False
        start_time = time.time()
        start_mem = get_memory_usage()
        
        story, critique = generate_story_idea(dna, db=db, multi_step=multi_step)
        end_time = time.time()
        end_mem = get_memory_usage()
        
        db.build_connections()
        eval_metrics = metrics_calc.evaluate_story(story, dna, db.graph)
        
        result = {
            "test_id": i,
            "multi_step": multi_step,
            "generation_time": end_time - start_time,
            "memory_delta": end_mem - start_mem,
            "critique_score": critique['overall_score'],
            "graph_density": eval_metrics['graph_density'],
            "coherence": eval_metrics['coherence'],
            "lotr_similarity": eval_metrics['classic_similarities']['LOTR'],
            "dune_similarity": eval_metrics['classic_similarities']['Dune'],
            "story_length": len(story),
        }
        results.append(result)
        print(f"Test {i}: Score {result['critique_score']:.2f}, Time {result['generation_time']:.2f}s, Mem {result['memory_delta']:.2f}MB")
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    scores = [r["critique_score"] for r in results]
    times = [r["generation_time"] for r in results]
    densities = [r["graph_density"] for r in results]
    
    summary = {
        "total_tests": len(results),
        "avg_score": sum(scores)/len(scores),
        "avg_time": sum(times)/len(times),
        "avg_density": sum(densities)/len(densities),
        "max_score": max(scores),
        "min_score": min(scores),
    }
    with open("test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Testing complete. Results saved.")

if __name__ == "__main__":
    run_tests(num_tests=5)  # Quick test