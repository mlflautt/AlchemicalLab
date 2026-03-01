#!/usr/bin/env python3
"""
AlchemicalLab CLI
=================

Command-line interface for interacting with AlchemicalLab generative systems.

Usage:
    python alchemical_cli.py <command> [options]

Commands:
    graph       - Knowledge graph operations
    ca          - Cellular automata simulations  
    story       - Story generation
    test        - Run test suite
    status      - Show system status
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_graph(args):
    """Knowledge graph operations."""
    from GraphEngine.core.knowledge_graph import KnowledgeGraph
    
    graph = KnowledgeGraph(
        db_path=args.db_path or "knowledge_graph.db",
        enable_vector_search=False
    )
    
    if args.action == "list":
        nodes = graph.list_nodes(node_type=args.type, limit=args.limit or 50)
        print(f"Total {args.type or 'node'}s: {len(nodes)}")
        for node in nodes:
            name = node.properties.get('name', node.id[:8])
            print(f"  - {node.id[:12]}... [{node.type}]: {name}")
            
    elif args.action == "stats":
        stats = graph.get_node_stats()
        print("Graph Statistics:")
        for node_type, count in sorted(stats.items()):
            print(f"  {node_type}: {count}")
        print(f"  Total: {sum(stats.values())}")
    
    elif args.action == "search":
        results = graph.search(args.query, node_types=args.types, limit=args.limit or 10)
        print(f"Found {len(results)} results for '{args.query}':")
        for node_id in results:
            node = graph.get_node(node_id)
            if node:
                name = node.properties.get('name', node_id[:8])
                print(f"  - {node_id[:12]}... [{node.type}]: {name}")
    
    elif args.action == "info":
        node = graph.get_node(args.entity_id)
        if not node:
            print(f"Entity not found: {args.entity_id}")
            return
        print(f"Entity: {node.id}")
        print(f"  Type: {node.type}")
        print(f"  Created: {node.created}")
        print(f"  Properties:")
        for k, v in node.properties.items():
            print(f"    {k}: {v}")
        edges = graph.get_edges(node.id)
        if edges:
            print(f"  Edges: {len(edges)}")
            for e in edges[:5]:
                print(f"    - {e.edge_type} -> {e.target_id[:12]}")
    
    elif args.action == "add":
        node_id = graph.add_node(
            node_type=args.type,
            properties=json.loads(args.properties) if args.properties else {"name": args.name},
            source="cli"
        )
        print(f"Created node: {node_id}")
    
    elif args.action == "export":
        data = graph.export_graph()
        output_file = args.output or "graph_export.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Exported to {output_file}")


def cmd_ca(args):
    """Cellular automata operations."""
    import numpy as np
    
    if args.action == "run":
        from CALab.hybrid_ca_aggregator import HybridCAAggregator
        
        size = tuple(map(int, args.size.split('x')))
        aggregator = HybridCAAggregator(world_size=size)
        aggregator.initialize(density=args.density, seed=args.seed or 42)
        
        print(f"Running CA simulation: {size[0]}x{size[1]}, density={args.density}")
        
        for i in range(args.steps):
            aggregator.step()
            if i % args.interval == 0:
                ws = aggregator.get_world_state()
                species_count = len(ws.species)
                print(f"  Gen {i}: {species_count} species, {ws.total_population} total pop")
        
        if args.save:
            ws = aggregator.get_world_state()
            with open(args.save, 'w') as f:
                f.write(ws.to_json())
            print(f"Saved to {args.save}")
    
    elif args.action == "visualize":
        from CALab.visual_ca import visualize_ca
        import matplotlib.pyplot as plt
        
        np.random.seed(args.seed or 42)
        grid = np.random.random((args.size, args.size)) < args.density
        
        for _ in range(args.steps):
            from CALab.traditional_ca import step_conway
            grid = step_conway(grid)
        
        visualize_ca(grid, title=f"CA after {args.steps} steps")
        plt.savefig(args.output or "ca_output.png")
        print(f"Saved visualization to {args.output or 'ca_output.png'}")
    
    elif args.action == "detect":
        from CALab.emergent_graphs import CAPatternDetector
        
        np.random.seed(args.seed or 42)
        grid = np.random.random((args.size, args.size)) < args.density
        
        detector = CAPatternDetector()
        patterns = detector.detect_patterns(grid, 0)
        
        print(f"Detected {len(patterns)} patterns:")
        for p in patterns:
            print(f"  - {p.pattern_type} at {p.center}, {len(p.cells)} cells")


def cmd_story(args):
    """Story generation operations."""
    from StoryLab.pipeline.generation_pipeline import GenerationPipeline
    
    if args.action == "generate":
        from GraphEngine.core.knowledge_graph import KnowledgeGraph
        from GraphEngine.context.extractor import ContextExtractor
        from StoryLab.agents.character_agent import CharacterAgent
        
        graph = KnowledgeGraph(enable_vector_search=False)
        
        if args.world_dna:
            with open(args.world_dna, 'r') as f:
                world_dna = f.read()
        
        pipeline = GenerationPipeline(graph)
        
        if args.entity:
            extractor = ContextExtractor(graph)
            context = extractor.get_context(args.entity, args.task or "scene_generation")
            print(f"Context for {args.entity}:")
            print(f"  Related entities: {len(context.related_entities)}")
            print(f"  Relationships: {len(context.relationships)}")
            print(f"  Token count: {context.token_count}")
    
    elif args.action == "simulate":
        from StoryLab.simulation.world_simulator import WorldSimulator
        from GraphEngine.core.knowledge_graph import KnowledgeGraph
        
        graph = KnowledgeGraph(enable_vector_search=False)
        
        graph.add_node("location", {"name": "Dark Forest", "type": "forest"})
        graph.add_node("location", {"name": "Castle", "type": "fortress"})
        graph.add_node("species", {"name": "Wolves", "population": 50})
        
        simulator = WorldSimulator(graph)
        
        print("Running world simulation...")
        for i in range(args.steps):
            result = simulator.step()
            print(f"  Gen {i}: {result['tensions_detected']} tensions, {len(result['events_generated'])} events")
        
        summary = simulator.get_world_state_summary()
        print(f"\nWorld State: {summary['events_generated']} total events generated")


def cmd_test(args):
    """Run tests."""
    if args.quick:
        print("Running quick tests...")
        os.system("python tests/test_cross_lab.py --quick")
    else:
        print("Running full test suite...")
        os.system("python tests/test_cross_lab.py")


def cmd_status(args):
    """Show system status."""
    print("=" * 50)
    print("AlchemicalLab Status")
    print("=" * 50)
    
    print("\nDependencies:")
    deps = [
        ("numpy", "numpy"),
        ("jax", "jax"),
        ("networkx", "networkx"),
        ("chromadb", "chromadb"),
    ]
    for name, import_name in deps:
        try:
            __import__(import_name)
            print(f"  {name}: OK")
        except ImportError:
            print(f"  {name}: MISSING")
    
    print("\nKnowledge Graph:")
    try:
        from GraphEngine.core.knowledge_graph import KnowledgeGraph
        graph = KnowledgeGraph(enable_vector_search=False)
        stats = graph.get_node_stats()
        total = sum(stats.values())
        print(f"  Nodes: {total}")
        for ntype, count in sorted(stats.items()):
            if count > 0:
                print(f"    - {ntype}: {count}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nCALab:")
    try:
        from CALab.hybrid_ca_aggregator import HybridCAAggregator
        print("  HybridCAAggregator: OK")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="AlchemicalLab CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    graph_parser = subparsers.add_parser("graph", help="Knowledge graph operations")
    graph_parser.add_argument("action", choices=["list", "stats", "search", "info", "add", "export"])
    graph_parser.add_argument("--type", help="Node type filter")
    graph_parser.add_argument("--db-path", help="Path to database")
    graph_parser.add_argument("--limit", type=int, help="Result limit")
    graph_parser.add_argument("--query", help="Search query")
    graph_parser.add_argument("--types", help="Comma-separated node types")
    graph_parser.add_argument("--entity-id", help="Entity ID for info command")
    graph_parser.add_argument("--name", help="Entity name for add command")
    graph_parser.add_argument("--properties", help="JSON properties for add command")
    graph_parser.add_argument("--output", help="Output file path")
    graph_parser.set_defaults(func=cmd_graph)
    
    ca_parser = subparsers.add_parser("ca", help="Cellular automata operations")
    ca_parser.add_argument("action", choices=["run", "visualize", "detect"])
    ca_parser.add_argument("--size", type=int, default=50, help="Grid size")
    ca_parser.add_argument("--steps", type=int, default=50, help="Number of steps")
    ca_parser.add_argument("--density", type=float, default=0.3, help="Initial density")
    ca_parser.add_argument("--seed", type=int, help="Random seed")
    ca_parser.add_argument("--interval", type=int, default=10, help="Progress interval")
    ca_parser.add_argument("--save", help="Save state to file")
    ca_parser.add_argument("--output", help="Output file path")
    ca_parser.set_defaults(func=cmd_ca)
    
    story_parser = subparsers.add_parser("story", help="Story generation")
    story_parser.add_argument("action", choices=["generate", "simulate"])
    story_parser.add_argument("--entity", help="Entity ID")
    story_parser.add_argument("--task", help="Task type")
    story_parser.add_argument("--world-dna", help="World DNA file")
    story_parser.add_argument("--steps", type=int, default=10, help="Simulation steps")
    story_parser.set_defaults(func=cmd_story)
    
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    test_parser.set_defaults(func=cmd_test)
    
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()