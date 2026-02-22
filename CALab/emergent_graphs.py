"""
Emergent Graph System from Cellular Automata
Patterns in CA become nodes, interactions become edges, creating emergent graph structures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time

plt.style.use('dark_background')

@dataclass
class CAPattern:
    """Represents a detected pattern in the CA."""
    pattern_type: str  # 'glider', 'oscillator', 'still_life', 'chaotic'
    center: Tuple[int, int]  # Center position
    size: int  # Approximate size
    period: int = 1  # Oscillation period (1 for static)
    birth_generation: int = 0
    last_seen: int = 0
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    connections: Set[int] = field(default_factory=set)  # Connected pattern IDs
    
@dataclass 
class GraphNode:
    """Graph node representing a CA pattern."""
    pattern_id: int
    pattern: CAPattern
    influence_radius: float = 10.0
    activity_level: float = 1.0
    node_type: str = "pattern"
    
@dataclass
class GraphEdge:
    """Graph edge representing interaction between patterns."""
    source_id: int
    target_id: int
    interaction_type: str  # 'proximity', 'collision', 'spawning', 'synchrony'
    strength: float = 1.0
    age: int = 0

class CAPatternDetector:
    """Detects patterns in CA grids."""
    
    def __init__(self):
        # Template patterns for detection
        self.templates = {
            'block': np.array([[1,1],[1,1]]),
            'blinker': np.array([[1,1,1]]),
            'glider': np.array([[0,1,0],[0,0,1],[1,1,1]]),
            'toad': np.array([[0,1,1,1],[1,1,1,0]])
        }
        self.pattern_history = defaultdict(list)
        
    def detect_patterns(self, grid: np.ndarray, generation: int) -> List[CAPattern]:
        """Detect patterns in current grid state."""
        patterns = []
        pattern_id = 0
        
        # Find connected components of live cells
        labeled_grid, num_features = self._label_components(grid)
        
        for component_id in range(1, num_features + 1):
            component_mask = (labeled_grid == component_id)
            component_coords = np.where(component_mask)
            
            if len(component_coords[0]) == 0:
                continue
                
            # Get bounding box and center
            min_y, max_y = component_coords[0].min(), component_coords[0].max()
            min_x, max_x = component_coords[1].min(), component_coords[1].max()
            center_y = (min_y + max_y) // 2
            center_x = (min_x + max_x) // 2
            size = max(max_y - min_y, max_x - min_x) + 1
            
            # Extract local pattern
            local_pattern = grid[min_y:max_y+1, min_x:max_x+1]
            
            # Classify pattern
            pattern_type = self._classify_pattern(local_pattern, (center_y, center_x))
            
            # Check if this continues an existing pattern trajectory
            period = self._estimate_period(pattern_type, (center_y, center_x), generation)
            
            pattern = CAPattern(
                pattern_type=pattern_type,
                center=(center_y, center_x),
                size=size,
                period=period,
                birth_generation=generation,
                last_seen=generation
            )
            
            patterns.append(pattern)
            pattern_id += 1
            
        return patterns
    
    def _label_components(self, grid: np.ndarray) -> Tuple[np.ndarray, int]:
        """Find connected components of live cells."""
        from scipy import ndimage
        
        # Define 8-connectivity
        structure = np.array([[1,1,1],[1,1,1],[1,1,1]])
        return ndimage.label(grid, structure=structure)
    
    def _classify_pattern(self, local_pattern: np.ndarray, center: Tuple[int, int]) -> str:
        """Classify the type of pattern."""
        total_cells = np.sum(local_pattern)
        height, width = local_pattern.shape
        
        # Simple heuristics for pattern classification
        if total_cells <= 4 and height <= 2 and width <= 2:
            return 'still_life'
        elif total_cells <= 6 and (height <= 2 or width <= 2):
            return 'oscillator'
        elif total_cells <= 5 and max(height, width) >= 3:
            return 'glider'
        elif total_cells > 20:
            return 'chaotic'
        else:
            return 'unknown'
    
    def _estimate_period(self, pattern_type: str, center: Tuple[int, int], generation: int) -> int:
        """Estimate oscillation period based on pattern history."""
        if pattern_type in ['still_life', 'chaotic']:
            return 1
        elif pattern_type == 'oscillator':
            return 2  # Most common period
        elif pattern_type == 'glider':
            return 4  # Glider period
        else:
            return 1

class EmergentGraphSystem:
    """System that creates graphs from CA patterns and influences CA evolution."""
    
    def __init__(self, ca_size: Tuple[int, int] = (100, 100)):
        self.ca_size = ca_size
        self.ca_grid = np.zeros(ca_size, dtype=bool)
        self.generation = 0
        
        # Pattern detection and tracking
        self.detector = CAPatternDetector()
        self.patterns: Dict[int, CAPattern] = {}
        self.pattern_id_counter = 0
        
        # Graph components
        self.graph_nodes: Dict[int, GraphNode] = {}
        self.graph_edges: Dict[Tuple[int, int], GraphEdge] = {}
        self.interaction_threshold = 15.0  # Distance for pattern interactions
        
        # Evolution tracking
        self.pattern_history = []
        self.graph_history = []
        
    def initialize_ca(self, density: float = 0.3, seed: int = 42):
        """Initialize CA with random configuration and some structured patterns."""
        np.random.seed(seed)
        
        # Random soup
        self.ca_grid = np.random.random(self.ca_size) < density * 0.3
        
        # Add some structured patterns
        self._add_seed_patterns()
        
    def _add_seed_patterns(self):
        """Add some seed patterns to bootstrap emergent behavior."""
        h, w = self.ca_size
        margin = 15
        
        # Glider
        glider = np.array([[0,1,0],[0,0,1],[1,1,1]], dtype=bool)
        self.ca_grid[margin:margin+3, margin:margin+3] = glider
        
        # R-pentomino
        r_pent = np.array([[0,1,1],[1,1,0],[0,1,0]], dtype=bool)
        self.ca_grid[h//3:h//3+3, w//3:w//3+3] = r_pent
        
        # Oscillators
        blinker = np.array([[1,1,1]], dtype=bool)
        self.ca_grid[h//4:h//4+1, w//4:w//4+3] = blinker
        
    def step_ca(self):
        """Execute one CA generation using Conway's Game of Life rules."""
        from scipy import ndimage
        
        # Count neighbors with wrap-around boundaries
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        padded = np.pad(self.ca_grid.astype(int), 1, mode='wrap')
        neighbors = ndimage.convolve(padded, kernel, mode='constant')[1:-1, 1:-1]
        
        # Conway rules
        new_grid = np.zeros_like(self.ca_grid)
        new_grid |= (~self.ca_grid) & (neighbors == 3)  # Birth
        new_grid |= self.ca_grid & ((neighbors == 2) | (neighbors == 3))  # Survival
        
        self.ca_grid = new_grid
        self.generation += 1
        
    def detect_and_update_patterns(self):
        """Detect patterns and update graph structure."""
        current_patterns = self.detector.detect_patterns(self.ca_grid, self.generation)
        
        # Update existing patterns and add new ones
        active_pattern_ids = set()
        
        for pattern in current_patterns:
            # Try to match with existing patterns
            matched_id = self._match_existing_pattern(pattern)
            
            if matched_id is not None:
                # Update existing pattern
                self.patterns[matched_id].last_seen = self.generation
                self.patterns[matched_id].trajectory.append(pattern.center)
                active_pattern_ids.add(matched_id)
            else:
                # New pattern
                pattern_id = self.pattern_id_counter
                self.pattern_id_counter += 1
                
                self.patterns[pattern_id] = pattern
                self.graph_nodes[pattern_id] = GraphNode(pattern_id, pattern)
                active_pattern_ids.add(pattern_id)
        
        # Remove old patterns
        dead_patterns = []
        for pid in list(self.patterns.keys()):
            if pid not in active_pattern_ids:
                if self.generation - self.patterns[pid].last_seen > 5:
                    dead_patterns.append(pid)
        
        for pid in dead_patterns:
            self._remove_pattern(pid)
        
        # Update graph edges based on pattern interactions
        self._update_graph_edges()
        
    def _match_existing_pattern(self, new_pattern: CAPattern) -> Optional[int]:
        """Try to match new pattern with existing ones."""
        match_distance = 8.0
        
        for pid, existing_pattern in self.patterns.items():
            if existing_pattern.pattern_type != new_pattern.pattern_type:
                continue
                
            # Check distance from last known position
            if len(existing_pattern.trajectory) > 0:
                last_pos = existing_pattern.trajectory[-1]
            else:
                last_pos = existing_pattern.center
                
            distance = np.sqrt((new_pattern.center[0] - last_pos[0])**2 + 
                             (new_pattern.center[1] - last_pos[1])**2)
            
            if distance <= match_distance:
                return pid
                
        return None
    
    def _remove_pattern(self, pattern_id: int):
        """Remove pattern and associated graph components."""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
        if pattern_id in self.graph_nodes:
            del self.graph_nodes[pattern_id]
            
        # Remove associated edges
        edges_to_remove = []
        for edge_key in self.graph_edges:
            if pattern_id in edge_key:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            del self.graph_edges[edge_key]
    
    def _update_graph_edges(self):
        """Update graph edges based on pattern proximity and interactions."""
        current_edges = set()
        
        pattern_ids = list(self.patterns.keys())
        
        for i, pid1 in enumerate(pattern_ids):
            for pid2 in pattern_ids[i+1:]:
                pattern1 = self.patterns[pid1]
                pattern2 = self.patterns[pid2]
                
                # Calculate distance
                distance = np.sqrt((pattern1.center[0] - pattern2.center[0])**2 + 
                                 (pattern1.center[1] - pattern2.center[1])**2)
                
                if distance <= self.interaction_threshold:
                    edge_key = (min(pid1, pid2), max(pid1, pid2))
                    current_edges.add(edge_key)
                    
                    if edge_key not in self.graph_edges:
                        # Determine interaction type
                        interaction_type = self._classify_interaction(pattern1, pattern2, distance)
                        
                        self.graph_edges[edge_key] = GraphEdge(
                            source_id=pid1,
                            target_id=pid2,
                            interaction_type=interaction_type,
                            strength=max(0.1, 1.0 - distance / self.interaction_threshold)
                        )
                    else:
                        # Update existing edge
                        self.graph_edges[edge_key].age += 1
                        self.graph_edges[edge_key].strength = max(0.1, 
                            1.0 - distance / self.interaction_threshold)
        
        # Remove edges that are no longer active
        edges_to_remove = []
        for edge_key in self.graph_edges:
            if edge_key not in current_edges:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            del self.graph_edges[edge_key]
    
    def _classify_interaction(self, pattern1: CAPattern, pattern2: CAPattern, distance: float) -> str:
        """Classify the type of interaction between patterns."""
        if distance <= 5.0:
            return 'collision'
        elif pattern1.pattern_type == pattern2.pattern_type:
            return 'synchrony'
        elif 'glider' in [pattern1.pattern_type, pattern2.pattern_type]:
            return 'approach'
        else:
            return 'proximity'
    
    def apply_graph_influence(self):
        """Apply graph structure influence back to CA evolution."""
        # Graph influences CA through local modifications
        influence_applied = False
        
        for edge in self.graph_edges.values():
            if edge.interaction_type == 'collision' and edge.strength > 0.8:
                # Strong collision - create disturbance
                node1 = self.graph_nodes[edge.source_id]
                node2 = self.graph_nodes[edge.target_id]
                
                center1 = node1.pattern.center
                center2 = node2.pattern.center
                
                # Create disturbance between patterns
                mid_y = (center1[0] + center2[0]) // 2
                mid_x = (center1[1] + center2[1]) // 2
                
                if 1 <= mid_y < self.ca_size[0]-1 and 1 <= mid_x < self.ca_size[1]-1:
                    # Add some noise/disturbance
                    self.ca_grid[mid_y-1:mid_y+2, mid_x-1:mid_x+2] = np.random.random((3,3)) < 0.5
                    influence_applied = True
        
        return influence_applied
    
    def get_graph_stats(self) -> Dict:
        """Get current graph statistics."""
        return {
            'num_nodes': len(self.graph_nodes),
            'num_edges': len(self.graph_edges),
            'pattern_types': {ptype: sum(1 for p in self.patterns.values() 
                                       if p.pattern_type == ptype) 
                            for ptype in ['glider', 'oscillator', 'still_life', 'chaotic', 'unknown']},
            'avg_connectivity': len(self.graph_edges) * 2 / max(1, len(self.graph_nodes)),
            'generation': self.generation
        }

class EmergentGraphVisualizer:
    """Visualizes CA with emergent graph overlay."""
    
    def __init__(self, system: EmergentGraphSystem):
        self.system = system
        
        # Setup dark theme
        plt.rcParams.update({
            'figure.facecolor': '#0a0a0a',
            'axes.facecolor': '#1a1a1a',
            'text.color': '#ffffff',
            'axes.labelcolor': '#ffffff',
        })
        
        self.fig, ((self.ax_ca, self.ax_graph), (self.ax_stats, self.ax_timeline)) = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # CA visualization
        self.ax_ca.set_facecolor('#1a1a1a')
        colors_ca = ['#0a0a0a', '#00ffff']
        self.cmap_ca = plt.matplotlib.colors.ListedColormap(colors_ca)
        
        # Graph visualization setup
        self.ax_graph.set_facecolor('#1a1a1a')
        
        # Statistics and timeline
        self.ax_stats.set_facecolor('#1a1a1a')
        self.ax_timeline.set_facecolor('#1a1a1a')
        
        # Data tracking
        self.stats_history = []
        
    def update_visualization(self):
        """Update all visualization components."""
        # Clear axes
        self.ax_ca.clear()
        self.ax_graph.clear()
        self.ax_stats.clear()
        self.ax_timeline.clear()
        
        # Set backgrounds
        for ax in [self.ax_ca, self.ax_graph, self.ax_stats, self.ax_timeline]:
            ax.set_facecolor('#1a1a1a')
        
        # 1. CA Grid with pattern overlay
        self._draw_ca_with_patterns()
        
        # 2. Graph representation
        self._draw_graph()
        
        # 3. Statistics
        self._draw_statistics()
        
        # 4. Timeline
        self._draw_timeline()
        
        plt.tight_layout()
    
    def _draw_ca_with_patterns(self):
        """Draw CA grid with detected patterns highlighted."""
        self.ax_ca.imshow(self.system.ca_grid, cmap=self.cmap_ca, interpolation='nearest')
        
        # Overlay detected patterns
        colors = {'glider': '#ff6b6b', 'oscillator': '#4ecdc4', 
                 'still_life': '#45b7d1', 'chaotic': '#96ceb4', 'unknown': '#feca57'}
        
        for pid, pattern in self.system.patterns.items():
            color = colors.get(pattern.pattern_type, '#ffffff')
            
            # Draw pattern boundary
            circle = Circle(pattern.center[::-1], radius=pattern.size/2 + 2, 
                          fill=False, color=color, linewidth=2, alpha=0.8)
            self.ax_ca.add_patch(circle)
            
            # Pattern ID label
            self.ax_ca.text(pattern.center[1], pattern.center[0], str(pid), 
                          color=color, fontsize=8, ha='center', va='center',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        self.ax_ca.set_title(f'CA Grid + Patterns (Gen {self.system.generation})', 
                           color='white', fontsize=12)
        self.ax_ca.set_xlim(-0.5, self.system.ca_size[1]-0.5)
        self.ax_ca.set_ylim(-0.5, self.system.ca_size[0]-0.5)
        
    def _draw_graph(self):
        """Draw the emergent graph structure."""
        if not self.system.graph_nodes:
            self.ax_graph.text(0.5, 0.5, 'No patterns detected', 
                             transform=self.ax_graph.transAxes, ha='center', va='center',
                             color='white', fontsize=12)
            self.ax_graph.set_title('Emergent Graph Structure', color='white')
            return
            
        # Create NetworkX graph for layout
        G = nx.Graph()
        
        # Add nodes
        for pid, node in self.system.graph_nodes.items():
            G.add_node(pid, pattern_type=node.pattern.pattern_type)
        
        # Add edges
        for (pid1, pid2), edge in self.system.graph_edges.items():
            G.add_edge(pid1, pid2, weight=edge.strength, type=edge.interaction_type)
        
        if len(G.nodes()) == 0:
            return
            
        # Layout
        try:
            pos = nx.spring_layout(G, k=1, iterations=50)
        except:
            pos = {pid: (0.5, 0.5) for pid in G.nodes()}
        
        # Draw nodes
        colors = {'glider': '#ff6b6b', 'oscillator': '#4ecdc4', 
                 'still_life': '#45b7d1', 'chaotic': '#96ceb4', 'unknown': '#feca57'}
        
        for pid in G.nodes():
            pattern_type = self.system.patterns[pid].pattern_type
            color = colors.get(pattern_type, '#ffffff')
            size = 300 + self.system.patterns[pid].size * 50
            
            self.ax_graph.scatter(pos[pid][0], pos[pid][1], 
                                s=size, c=color, alpha=0.8, edgecolors='white', linewidth=2)
            
            # Node labels
            self.ax_graph.text(pos[pid][0], pos[pid][1], str(pid), 
                             color='black', fontsize=10, ha='center', va='center', weight='bold')
        
        # Draw edges
        edge_colors = {'collision': '#ff4757', 'synchrony': '#5352ed', 
                      'approach': '#ff6348', 'proximity': '#747d8c'}
        
        for (pid1, pid2), edge_data in self.system.graph_edges.items():
            color = edge_colors.get(edge_data.interaction_type, '#ffffff')
            width = edge_data.strength * 3
            alpha = min(1.0, edge_data.strength + 0.3)
            
            x_coords = [pos[pid1][0], pos[pid2][0]]
            y_coords = [pos[pid1][1], pos[pid2][1]]
            
            self.ax_graph.plot(x_coords, y_coords, color=color, 
                             linewidth=width, alpha=alpha)
        
        self.ax_graph.set_title('Emergent Graph Structure', color='white', fontsize=12)
        self.ax_graph.set_xlim(-1.2, 1.2)
        self.ax_graph.set_ylim(-1.2, 1.2)
        self.ax_graph.axis('off')
        
    def _draw_statistics(self):
        """Draw current statistics."""
        stats = self.system.get_graph_stats()
        self.stats_history.append(stats)
        
        # Text statistics
        stats_text = f"Generation: {stats['generation']}\\n"
        stats_text += f"Graph Nodes: {stats['num_nodes']}\\n"
        stats_text += f"Graph Edges: {stats['num_edges']}\\n"
        stats_text += f"Connectivity: {stats['avg_connectivity']:.2f}\\n\\n"
        stats_text += "Pattern Types:\\n"
        
        for ptype, count in stats['pattern_types'].items():
            if count > 0:
                stats_text += f"  {ptype}: {count}\\n"
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          verticalalignment='top', fontsize=11, color='white',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#333333', alpha=0.9))
        
        self.ax_stats.set_title('System Statistics', color='white', fontsize=12)
        self.ax_stats.axis('off')
        
    def _draw_timeline(self):
        """Draw timeline of graph evolution."""
        if len(self.stats_history) < 2:
            return
            
        generations = [s['generation'] for s in self.stats_history]
        node_counts = [s['num_nodes'] for s in self.stats_history]
        edge_counts = [s['num_edges'] for s in self.stats_history]
        
        self.ax_timeline.plot(generations, node_counts, '#00ff88', linewidth=2, label='Nodes', marker='o', markersize=3)
        self.ax_timeline.plot(generations, edge_counts, '#ff6b6b', linewidth=2, label='Edges', marker='s', markersize=3)
        
        self.ax_timeline.set_xlabel('Generation', color='white')
        self.ax_timeline.set_ylabel('Count', color='white')
        self.ax_timeline.set_title('Graph Evolution Timeline', color='white', fontsize=12)
        self.ax_timeline.legend(facecolor='#333333', edgecolor='white')
        self.ax_timeline.grid(True, color='#333333', alpha=0.3)
        
        # Style axes
        self.ax_timeline.tick_params(colors='white')
        for spine in self.ax_timeline.spines.values():
            spine.set_color('#555555')
    
    def run_demo(self, max_generations: int = 200, steps_per_second: float = 8):
        """Run the emergent graph demo."""
        generation_count = 0
        
        def animate(frame):
            nonlocal generation_count
            if generation_count < max_generations:
                # Evolution step
                self.system.step_ca()
                self.system.detect_and_update_patterns()
                
                # Occasionally apply graph influence
                if generation_count % 20 == 0:
                    influenced = self.system.apply_graph_influence()
                    if influenced:
                        print(f"Gen {self.system.generation}: Graph influenced CA")
                
                # Update visualization
                self.update_visualization()
                
                # Progress logging
                if generation_count % 25 == 0:
                    stats = self.system.get_graph_stats()
                    print(f"Gen {stats['generation']}: {stats['num_nodes']} patterns, "
                          f"{stats['num_edges']} connections")
                
                generation_count += 1
            
            return []
        
        interval = int(1000 / steps_per_second)
        ani = animation.FuncAnimation(self.fig, animate, interval=interval, 
                                    blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
        return ani

def main():
    """Demo of emergent graph system."""
    print("Emergent Graph System from Cellular Automata")
    print("=" * 50)
    
    # Create system
    system = EmergentGraphSystem(ca_size=(80, 80))
    system.initialize_ca(density=0.25, seed=42)
    
    # Create visualizer
    viz = EmergentGraphVisualizer(system)
    
    print("System initialized:")
    print("- CA patterns become graph nodes")
    print("- Pattern interactions become edges") 
    print("- Graph structure influences CA evolution")
    print("- Watch for emergent network behaviors")
    print()
    
    try:
        print("Starting emergent graph demonstration...")
        viz.run_demo(max_generations=150, steps_per_second=6)
    except KeyboardInterrupt:
        print("\\nDemo stopped by user")
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    main()