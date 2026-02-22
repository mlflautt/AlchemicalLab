"""
Multi-Scale Emergent Graph System
CA evolves at one timescale while graph structures emerge and evolve at different scales
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch, ConnectionPatch
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import random

plt.style.use('dark_background')

@dataclass
class MetaPattern:
    """Higher-order pattern composed of multiple CA patterns."""
    pattern_ids: Set[int]
    formation_type: str  # 'cluster', 'chain', 'oscillation', 'collision_cascade'
    stability: float = 0.0
    age: int = 0
    centroid: Tuple[float, float] = (0.0, 0.0)
    influence_strength: float = 1.0

@dataclass
class GraphMemory:
    """Memory system for graph evolution."""
    pattern_lifetimes: Dict[int, int] = field(default_factory=dict)
    interaction_history: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)
    meta_pattern_library: Dict[str, MetaPattern] = field(default_factory=dict)
    global_phase: str = "exploration"  # exploration, consolidation, reorganization

class MultiScaleEmergentSystem:
    """System with multiple temporal scales for CA and graph evolution."""
    
    def __init__(self, ca_size: Tuple[int, int] = (100, 100)):
        self.ca_size = ca_size
        self.ca_grid = np.zeros(ca_size, dtype=bool)
        self.generation = 0
        
        # Multi-scale timing
        self.ca_step = 1  # CA evolves every step
        self.pattern_detection_step = 3  # Pattern detection every 3 steps
        self.graph_evolution_step = 5  # Graph structure evolution every 5 steps
        self.meta_pattern_step = 15  # Meta-pattern analysis every 15 steps
        self.global_influence_step = 25  # Global influence every 25 steps
        
        # Pattern detection and tracking
        from emergent_graphs import CAPatternDetector
        self.detector = CAPatternDetector()
        self.patterns: Dict[int, 'CAPattern'] = {}
        self.pattern_id_counter = 0
        
        # Graph components with memory
        self.graph_nodes: Dict[int, 'GraphNode'] = {}
        self.graph_edges: Dict[Tuple[int, int], 'GraphEdge'] = {}
        self.meta_patterns: Dict[int, MetaPattern] = {}
        self.memory = GraphMemory()
        
        # Multi-scale parameters
        self.interaction_threshold = 20.0
        self.meta_pattern_threshold = 8.0
        self.global_connectivity = 0.0
        self.system_entropy = 0.0
        
        # Evolution history for different scales
        self.ca_history = deque(maxlen=50)
        self.graph_history = deque(maxlen=200)
        self.meta_history = deque(maxlen=100)
        
    def initialize_ca(self, density: float = 0.3, seed: int = 42):
        """Initialize CA with structured seeding."""
        np.random.seed(seed)
        
        # Create zones with different dynamics
        h, w = self.ca_size
        
        # Zone 1: Random soup (top-left)
        self.ca_grid[:h//3, :w//3] = np.random.random((h//3, w//3)) < density * 0.4
        
        # Zone 2: Structured patterns (center)
        self._add_structured_seed(h//2, w//2)
        
        # Zone 3: Oscillator garden (bottom-right)
        self._add_oscillator_garden(2*h//3, 2*w//3)
        
    def _add_structured_seed(self, center_y: int, center_x: int):
        """Add structured patterns in center region."""
        patterns = {
            'glider': np.array([[0,1,0],[0,0,1],[1,1,1]]),
            'r_pent': np.array([[0,1,1],[1,1,0],[0,1,0]]),
            'pulsar_part': np.array([[1,1,1,0,0,0,1,1,1]])
        }
        
        offsets = [(-10, -10), (0, 0), (10, 10), (-5, 15), (15, -5)]
        pattern_names = list(patterns.keys())
        
        for i, (dy, dx) in enumerate(offsets):
            if i < len(pattern_names):
                pattern = patterns[pattern_names[i]]
                ph, pw = pattern.shape
                y, x = center_y + dy, center_x + dx
                
                if 0 <= y < self.ca_size[0]-ph and 0 <= x < self.ca_size[1]-pw:
                    self.ca_grid[y:y+ph, x:x+pw] = pattern
    
    def _add_oscillator_garden(self, center_y: int, center_x: int):
        """Add multiple small oscillators."""
        oscillators = {
            'blinker': np.array([[1,1,1]]),
            'toad': np.array([[0,1,1,1],[1,1,1,0]]),
            'beacon': np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])
        }
        
        positions = [(0, 0), (-8, 8), (8, -8), (0, 15), (15, 0)]
        
        for i, (dy, dx) in enumerate(positions):
            osc_name = list(oscillators.keys())[i % len(oscillators)]
            osc = oscillators[osc_name]
            oh, ow = osc.shape
            y, x = center_y + dy, center_x + dx
            
            if 0 <= y < self.ca_size[0]-oh and 0 <= x < self.ca_size[1]-ow:
                self.ca_grid[y:y+oh, x:x+ow] = osc
    
    def step_ca(self):
        """Step CA evolution."""
        from scipy import ndimage
        
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        padded = np.pad(self.ca_grid.astype(int), 1, mode='wrap')
        neighbors = ndimage.convolve(padded, kernel, mode='constant')[1:-1, 1:-1]
        
        new_grid = np.zeros_like(self.ca_grid)
        new_grid |= (~self.ca_grid) & (neighbors == 3)
        new_grid |= self.ca_grid & ((neighbors == 2) | (neighbors == 3))
        
        self.ca_grid = new_grid
        self.generation += 1
        
        # Store CA state history
        self.ca_history.append({
            'generation': self.generation,
            'alive_count': int(np.sum(self.ca_grid)),
            'density': float(np.mean(self.ca_grid))
        })
    
    def detect_patterns_multiscale(self):
        """Multi-scale pattern detection."""
        if self.generation % self.pattern_detection_step == 0:
            # Import pattern class
            from emergent_graphs import CAPattern, GraphNode
            
            current_patterns = self.detector.detect_patterns(self.ca_grid, self.generation)
            
            # Update pattern tracking with lifetime memory
            active_pattern_ids = set()
            
            for pattern in current_patterns:
                matched_id = self._match_existing_pattern(pattern)
                
                if matched_id is not None:
                    self.patterns[matched_id].last_seen = self.generation
                    self.patterns[matched_id].trajectory.append(pattern.center)
                    active_pattern_ids.add(matched_id)
                    
                    # Update lifetime memory
                    if matched_id in self.memory.pattern_lifetimes:
                        self.memory.pattern_lifetimes[matched_id] += 1
                    else:
                        self.memory.pattern_lifetimes[matched_id] = 1
                        
                else:
                    # New pattern
                    pattern_id = self.pattern_id_counter
                    self.pattern_id_counter += 1
                    
                    self.patterns[pattern_id] = pattern
                    self.graph_nodes[pattern_id] = GraphNode(pattern_id, pattern)
                    active_pattern_ids.add(pattern_id)
                    self.memory.pattern_lifetimes[pattern_id] = 1
            
            # Clean up old patterns
            dead_patterns = []
            for pid in list(self.patterns.keys()):
                if pid not in active_pattern_ids:
                    if self.generation - self.patterns[pid].last_seen > 8:
                        dead_patterns.append(pid)
            
            for pid in dead_patterns:
                self._remove_pattern(pid)
    
    def evolve_graph_structure(self):
        """Evolve graph structure at medium timescale."""
        if self.generation % self.graph_evolution_step == 0:
            self._update_graph_edges()
            self._calculate_global_metrics()
            
            # Store graph evolution history
            stats = self.get_multiscale_stats()
            self.graph_history.append(stats)
    
    def analyze_meta_patterns(self):
        """Analyze meta-patterns at slow timescale."""
        if self.generation % self.meta_pattern_step == 0:
            self._detect_meta_patterns()
            self._update_system_phase()
            
            # Store meta-pattern evolution
            meta_stats = {
                'generation': self.generation,
                'num_meta_patterns': len(self.meta_patterns),
                'global_phase': self.memory.global_phase,
                'system_entropy': self.system_entropy
            }
            self.meta_history.append(meta_stats)
    
    def apply_global_influence(self):
        """Apply global influence at slowest timescale."""
        if self.generation % self.global_influence_step == 0:
            influence_applied = False
            
            # Phase-dependent global influences
            if self.memory.global_phase == "exploration":
                influence_applied = self._encourage_diversity()
            elif self.memory.global_phase == "consolidation":
                influence_applied = self._reinforce_stable_patterns()
            elif self.memory.global_phase == "reorganization":
                influence_applied = self._trigger_reorganization()
            
            return influence_applied
        return False
    
    def _match_existing_pattern(self, new_pattern):
        """Match patterns with improved distance metric."""
        match_distance = 12.0
        
        for pid, existing_pattern in self.patterns.items():
            if existing_pattern.pattern_type != new_pattern.pattern_type:
                continue
            
            last_pos = existing_pattern.trajectory[-1] if existing_pattern.trajectory else existing_pattern.center
            distance = np.sqrt((new_pattern.center[0] - last_pos[0])**2 + (new_pattern.center[1] - last_pos[1])**2)
            
            # Adjust match distance based on pattern lifetime
            lifetime_factor = min(2.0, self.memory.pattern_lifetimes.get(pid, 1) / 10.0)
            adjusted_distance = match_distance * lifetime_factor
            
            if distance <= adjusted_distance:
                return pid
        return None
    
    def _remove_pattern(self, pattern_id: int):
        """Remove pattern and clean up."""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
        if pattern_id in self.graph_nodes:
            del self.graph_nodes[pattern_id]
        if pattern_id in self.memory.pattern_lifetimes:
            del self.memory.pattern_lifetimes[pattern_id]
        
        # Clean edges
        edges_to_remove = [key for key in self.graph_edges if pattern_id in key]
        for key in edges_to_remove:
            del self.graph_edges[key]
    
    def _update_graph_edges(self):
        """Update graph with interaction history."""
        from emergent_graphs import GraphEdge
        
        current_edges = set()
        pattern_ids = list(self.patterns.keys())
        
        for i, pid1 in enumerate(pattern_ids):
            for pid2 in pattern_ids[i+1:]:
                pattern1 = self.patterns[pid1]
                pattern2 = self.patterns[pid2]
                
                distance = np.sqrt((pattern1.center[0] - pattern2.center[0])**2 + (pattern1.center[1] - pattern2.center[1])**2)
                
                if distance <= self.interaction_threshold:
                    edge_key = (min(pid1, pid2), max(pid1, pid2))
                    current_edges.add(edge_key)
                    
                    strength = max(0.1, 1.0 - distance / self.interaction_threshold)
                    
                    # Update interaction history
                    if edge_key not in self.memory.interaction_history:
                        self.memory.interaction_history[edge_key] = []
                    self.memory.interaction_history[edge_key].append(strength)
                    
                    # Keep only recent history
                    if len(self.memory.interaction_history[edge_key]) > 20:
                        self.memory.interaction_history[edge_key] = self.memory.interaction_history[edge_key][-20:]
                    
                    if edge_key not in self.graph_edges:
                        interaction_type = self._classify_interaction_advanced(pattern1, pattern2, distance)
                        self.graph_edges[edge_key] = GraphEdge(pid1, pid2, interaction_type, strength)
                    else:
                        self.graph_edges[edge_key].strength = strength
                        self.graph_edges[edge_key].age += 1
        
        # Remove old edges
        edges_to_remove = [key for key in self.graph_edges if key not in current_edges]
        for key in edges_to_remove:
            del self.graph_edges[key]
    
    def _classify_interaction_advanced(self, pattern1, pattern2, distance):
        """Advanced interaction classification."""
        edge_key = (min(pattern1.birth_generation, pattern2.birth_generation), 
                   max(pattern1.birth_generation, pattern2.birth_generation))
        
        if distance <= 6.0:
            return 'collision'
        elif pattern1.pattern_type == pattern2.pattern_type:
            return 'resonance'
        elif 'glider' in [pattern1.pattern_type, pattern2.pattern_type] and 'oscillator' in [pattern1.pattern_type, pattern2.pattern_type]:
            return 'capture'
        elif distance <= 12.0:
            return 'influence'
        else:
            return 'proximity'
    
    def _detect_meta_patterns(self):
        """Detect higher-order patterns in graph structure."""
        if len(self.patterns) < 3:
            return
        
        # Detect clusters
        clusters = self._find_pattern_clusters()
        
        # Detect chains (patterns in sequence)
        chains = self._find_pattern_chains()
        
        # Update meta-patterns
        meta_id = 0
        self.meta_patterns = {}
        
        for cluster in clusters:
            if len(cluster) >= 3:
                centroid = self._calculate_centroid(cluster)
                stability = self._calculate_cluster_stability(cluster)
                
                self.meta_patterns[meta_id] = MetaPattern(
                    pattern_ids=set(cluster),
                    formation_type='cluster',
                    stability=stability,
                    centroid=centroid,
                    age=1
                )
                meta_id += 1
        
        for chain in chains:
            if len(chain) >= 3:
                centroid = self._calculate_centroid(chain)
                self.meta_patterns[meta_id] = MetaPattern(
                    pattern_ids=set(chain),
                    formation_type='chain',
                    stability=0.7,
                    centroid=centroid,
                    age=1
                )
                meta_id += 1
    
    def _find_pattern_clusters(self):
        """Find spatial clusters of patterns."""
        pattern_positions = [(pid, self.patterns[pid].center) for pid in self.patterns]
        clusters = []
        
        visited = set()
        for pid, pos in pattern_positions:
            if pid in visited:
                continue
            
            cluster = []
            to_visit = [(pid, pos)]
            
            while to_visit:
                current_pid, current_pos = to_visit.pop(0)
                if current_pid in visited:
                    continue
                
                visited.add(current_pid)
                cluster.append(current_pid)
                
                # Find nearby patterns
                for other_pid, other_pos in pattern_positions:
                    if other_pid not in visited:
                        distance = np.sqrt((current_pos[0] - other_pos[0])**2 + (current_pos[1] - other_pos[1])**2)
                        if distance <= self.meta_pattern_threshold:
                            to_visit.append((other_pid, other_pos))
            
            if len(cluster) >= 2:
                clusters.append(cluster)
        
        return clusters
    
    def _find_pattern_chains(self):
        """Find chains of connected patterns."""
        # Build adjacency for chain detection
        adjacency = defaultdict(list)
        for (pid1, pid2) in self.graph_edges:
            adjacency[pid1].append(pid2)
            adjacency[pid2].append(pid1)
        
        chains = []
        visited = set()
        
        for start_pid in self.patterns:
            if start_pid in visited or len(adjacency[start_pid]) != 1:
                continue  # Skip if not a chain end
            
            # Follow the chain
            chain = [start_pid]
            current = start_pid
            visited.add(current)
            
            while True:
                neighbors = [n for n in adjacency[current] if n not in visited]
                if len(neighbors) != 1:
                    break
                
                next_node = neighbors[0]
                chain.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if len(chain) >= 3:
                chains.append(chain)
        
        return chains
    
    def _calculate_centroid(self, pattern_ids):
        """Calculate centroid of pattern group."""
        positions = [self.patterns[pid].center for pid in pattern_ids]
        mean_y = np.mean([pos[0] for pos in positions])
        mean_x = np.mean([pos[1] for pos in positions])
        return (mean_y, mean_x)
    
    def _calculate_cluster_stability(self, cluster):
        """Calculate stability of pattern cluster."""
        if len(cluster) < 2:
            return 0.0
        
        # Based on lifetime variance and interaction strength
        lifetimes = [self.memory.pattern_lifetimes.get(pid, 1) for pid in cluster]
        lifetime_stability = 1.0 / (1.0 + np.var(lifetimes))
        
        # Interaction strength between cluster members
        interaction_strength = 0.0
        count = 0
        for i, pid1 in enumerate(cluster):
            for pid2 in cluster[i+1:]:
                edge_key = (min(pid1, pid2), max(pid1, pid2))
                if edge_key in self.graph_edges:
                    interaction_strength += self.graph_edges[edge_key].strength
                    count += 1
        
        avg_interaction = interaction_strength / max(1, count)
        return (lifetime_stability + avg_interaction) / 2.0
    
    def _calculate_global_metrics(self):
        """Calculate global connectivity and entropy."""
        if not self.patterns:
            self.global_connectivity = 0.0
            self.system_entropy = 0.0
            return
        
        num_patterns = len(self.patterns)
        num_edges = len(self.graph_edges)
        max_edges = num_patterns * (num_patterns - 1) // 2
        
        self.global_connectivity = num_edges / max(1, max_edges)
        
        # System entropy based on pattern type distribution
        pattern_types = [p.pattern_type for p in self.patterns.values()]
        type_counts = {}
        for ptype in pattern_types:
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        total = len(pattern_types)
        entropy = 0.0
        for count in type_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
        
        self.system_entropy = entropy
    
    def _update_system_phase(self):
        """Update global system phase based on metrics."""
        if len(self.graph_history) < 10:
            return
        
        recent_connectivity = [s['global_connectivity'] for s in list(self.graph_history)[-10:]]
        connectivity_trend = np.polyfit(range(len(recent_connectivity)), recent_connectivity, 1)[0]
        
        if connectivity_trend > 0.01:
            self.memory.global_phase = "exploration"
        elif abs(connectivity_trend) <= 0.01:
            self.memory.global_phase = "consolidation"
        else:
            self.memory.global_phase = "reorganization"
    
    def _encourage_diversity(self):
        """Encourage pattern diversity during exploration phase."""
        # Add random perturbations in low-density areas
        if np.mean(self.ca_grid) < 0.15:
            h, w = self.ca_size
            for _ in range(3):
                y, x = random.randint(10, h-10), random.randint(10, w-10)
                if np.sum(self.ca_grid[y-5:y+5, x-5:x+5]) < 5:
                    # Add small random pattern
                    self.ca_grid[y-1:y+2, x-1:x+2] = np.random.random((3, 3)) < 0.6
            return True
        return False
    
    def _reinforce_stable_patterns(self):
        """Reinforce stable patterns during consolidation."""
        stable_patterns = [(pid, p) for pid, p in self.patterns.items() 
                          if self.memory.pattern_lifetimes.get(pid, 0) > 15]
        
        if stable_patterns:
            # Protect stable patterns from disruption
            for pid, pattern in stable_patterns:
                y, x = pattern.center
                radius = pattern.size + 2
                
                # Clear nearby chaos
                y_min, y_max = max(0, y-radius), min(self.ca_size[0], y+radius+1)
                x_min, x_max = max(0, x-radius), min(self.ca_size[1], x+radius+1)
                
                local_region = self.ca_grid[y_min:y_max, x_min:x_max]
                if np.sum(local_region) > pattern.size * 3:  # Too chaotic
                    # Reduce local density slightly
                    noise_mask = np.random.random(local_region.shape) < 0.1
                    self.ca_grid[y_min:y_max, x_min:x_max] &= ~noise_mask
                    return True
        return False
    
    def _trigger_reorganization(self):
        """Trigger system reorganization."""
        # Create connections between distant pattern clusters
        meta_clusters = [mp for mp in self.meta_patterns.values() if mp.formation_type == 'cluster']
        
        if len(meta_clusters) >= 2:
            cluster1 = random.choice(meta_clusters)
            cluster2 = random.choice([c for c in meta_clusters if c != cluster1])
            
            # Create pathway between clusters
            y1, x1 = cluster1.centroid
            y2, x2 = cluster2.centroid
            
            # Linear interpolation pathway
            steps = max(abs(int(y2-y1)), abs(int(x2-x1)))
            if steps > 0:
                for i in range(1, steps):
                    y = int(y1 + (y2-y1) * i / steps)
                    x = int(x1 + (x2-x1) * i / steps)
                    if 0 <= y < self.ca_size[0] and 0 <= x < self.ca_size[1]:
                        if random.random() < 0.3:
                            self.ca_grid[y, x] = True
                return True
        return False
    
    def get_multiscale_stats(self):
        """Get comprehensive system statistics."""
        return {
            'generation': self.generation,
            'num_patterns': len(self.patterns),
            'num_edges': len(self.graph_edges),
            'num_meta_patterns': len(self.meta_patterns),
            'global_connectivity': self.global_connectivity,
            'system_entropy': self.system_entropy,
            'global_phase': self.memory.global_phase,
            'pattern_types': {ptype: sum(1 for p in self.patterns.values() if p.pattern_type == ptype) 
                            for ptype in ['glider', 'oscillator', 'still_life', 'chaotic', 'unknown']}
        }

class MultiScaleVisualizer:
    """Visualizer for multi-scale emergent system."""
    
    def __init__(self, system: MultiScaleEmergentSystem):
        self.system = system
        
        plt.rcParams.update({
            'figure.facecolor': '#0a0a0a',
            'axes.facecolor': '#1a1a1a',
            'text.color': '#ffffff',
            'axes.labelcolor': '#ffffff',
        })
        
        self.fig = plt.figure(figsize=(20, 12))
        
        # Create complex layout
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        self.ax_ca = self.fig.add_subplot(gs[0:2, 0:2])  # Large CA display
        self.ax_graph = self.fig.add_subplot(gs[0, 2])   # Graph structure
        self.ax_meta = self.fig.add_subplot(gs[1, 2])    # Meta patterns
        self.ax_phases = self.fig.add_subplot(gs[0, 3])  # Phase evolution
        self.ax_scales = self.fig.add_subplot(gs[1, 3])  # Multi-scale metrics
        self.ax_timeline = self.fig.add_subplot(gs[2, :]) # Timeline
        
        self.fig.patch.set_facecolor('#0a0a0a')
        for ax in [self.ax_ca, self.ax_graph, self.ax_meta, self.ax_phases, self.ax_scales, self.ax_timeline]:
            ax.set_facecolor('#1a1a1a')
        
        # Colors
        self.cmap_ca = plt.matplotlib.colors.ListedColormap(['#0a0a0a', '#00ffff'])
        
    def update_visualization(self):
        """Update all visualization components."""
        # Clear axes
        for ax in [self.ax_ca, self.ax_graph, self.ax_meta, self.ax_phases, self.ax_scales, self.ax_timeline]:
            ax.clear()
            ax.set_facecolor('#1a1a1a')
        
        self._draw_ca_multiscale()
        self._draw_graph_structure()
        self._draw_meta_patterns()
        self._draw_phase_evolution()
        self._draw_scale_metrics()
        self._draw_multiscale_timeline()
        
    def _draw_ca_multiscale(self):
        """Draw CA with multi-scale overlays."""
        self.ax_ca.imshow(self.system.ca_grid, cmap=self.cmap_ca, interpolation='nearest')
        
        # Pattern overlays with different scales
        pattern_colors = {'glider': '#ff6b6b', 'oscillator': '#4ecdc4', 
                         'still_life': '#45b7d1', 'chaotic': '#96ceb4', 'unknown': '#feca57'}
        
        # Individual patterns
        for pid, pattern in self.system.patterns.items():
            color = pattern_colors.get(pattern.pattern_type, '#ffffff')
            lifetime = self.system.memory.pattern_lifetimes.get(pid, 1)
            
            # Size based on lifetime
            radius = pattern.size/2 + 1 + min(5, lifetime/5)
            alpha = min(1.0, 0.3 + lifetime/20)
            
            circle = Circle(pattern.center[::-1], radius=radius, 
                          fill=False, color=color, linewidth=2, alpha=alpha)
            self.ax_ca.add_patch(circle)
        
        # Meta-pattern overlays
        meta_colors = {'cluster': '#ff9ff3', 'chain': '#54a0ff'}
        for meta_id, meta_pattern in self.system.meta_patterns.items():
            if meta_pattern.formation_type in meta_colors:
                color = meta_colors[meta_pattern.formation_type]
                
                # Draw convex hull around pattern group
                positions = [self.system.patterns[pid].center for pid in meta_pattern.pattern_ids 
                            if pid in self.system.patterns]
                
                if len(positions) >= 3:
                    positions = [(pos[1], pos[0]) for pos in positions]  # Swap for plotting
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(positions)
                        for simplex in hull.simplices:
                            self.ax_ca.plot([positions[simplex[0]][0], positions[simplex[1]][0]], 
                                          [positions[simplex[0]][1], positions[simplex[1]][1]], 
                                          color=color, linewidth=3, alpha=0.6)
                    except:
                        pass  # Skip if ConvexHull fails
        
        phase_color = {'exploration': '#ff6348', 'consolidation': '#2ed573', 'reorganization': '#5352ed'}
        current_color = phase_color.get(self.system.memory.global_phase, '#ffffff')
        
        self.ax_ca.set_title(f'Multi-Scale CA (Gen {self.system.generation}) - Phase: {self.system.memory.global_phase}', 
                           color=current_color, fontsize=14)
        
    def _draw_graph_structure(self):
        """Draw current graph structure."""
        if not self.system.graph_nodes:
            self.ax_graph.text(0.5, 0.5, 'No patterns', transform=self.ax_graph.transAxes, 
                             ha='center', va='center', color='white')
            return
        
        G = nx.Graph()
        for pid in self.system.graph_nodes:
            G.add_node(pid)
        for (pid1, pid2) in self.system.graph_edges:
            G.add_edge(pid1, pid2)
        
        if len(G.nodes()) > 0:
            try:
                pos = nx.spring_layout(G, k=0.8, iterations=30)
            except:
                pos = {pid: (random.random(), random.random()) for pid in G.nodes()}
            
            # Draw nodes colored by lifetime
            for pid in G.nodes():
                lifetime = self.system.memory.pattern_lifetimes.get(pid, 1)
                color_intensity = min(1.0, lifetime / 20)
                color = plt.cm.plasma(color_intensity)
                size = 100 + lifetime * 10
                
                self.ax_graph.scatter(pos[pid][0], pos[pid][1], s=size, c=[color], alpha=0.8)
            
            # Draw edges
            for (pid1, pid2), edge in self.system.graph_edges.items():
                x_coords = [pos[pid1][0], pos[pid2][0]]
                y_coords = [pos[pid1][1], pos[pid2][1]]
                alpha = edge.strength * 0.8
                
                self.ax_graph.plot(x_coords, y_coords, color='white', alpha=alpha, linewidth=edge.strength*2)
        
        self.ax_graph.set_title(f'Graph Structure\\n{len(self.system.graph_nodes)} nodes', color='white', fontsize=10)
        self.ax_graph.axis('off')
    
    def _draw_meta_patterns(self):
        """Draw meta-pattern analysis."""
        meta_types = {}
        for meta in self.system.meta_patterns.values():
            meta_types[meta.formation_type] = meta_types.get(meta.formation_type, 0) + 1
        
        if meta_types:
            types = list(meta_types.keys())
            counts = list(meta_types.values())
            colors = ['#ff9ff3', '#54a0ff', '#ff6b6b', '#4ecdc4'][:len(types)]
            
            bars = self.ax_meta.bar(types, counts, color=colors, alpha=0.8)
            self.ax_meta.set_ylabel('Count', color='white')
            self.ax_meta.tick_params(colors='white')
        
        self.ax_meta.set_title(f'Meta-Patterns\\n{len(self.system.meta_patterns)} total', color='white', fontsize=10)
    
    def _draw_phase_evolution(self):
        """Draw phase evolution over time."""
        if len(self.system.meta_history) > 1:
            generations = [h['generation'] for h in self.system.meta_history]
            phases = [h['global_phase'] for h in self.system.meta_history]
            
            phase_map = {'exploration': 0, 'consolidation': 1, 'reorganization': 2}
            phase_values = [phase_map.get(p, 0) for p in phases]
            
            self.ax_phases.plot(generations, phase_values, 'o-', color='#ff6b6b', linewidth=2, markersize=4)
            self.ax_phases.set_yticks([0, 1, 2])
            self.ax_phases.set_yticklabels(['Explore', 'Consolidate', 'Reorganize'], fontsize=8)
            self.ax_phases.set_ylabel('Phase', color='white')
            self.ax_phases.tick_params(colors='white')
        
        self.ax_phases.set_title('System Phase', color='white', fontsize=10)
    
    def _draw_scale_metrics(self):
        """Draw multi-scale metrics."""
        if len(self.system.graph_history) > 1:
            generations = [h['generation'] for h in self.system.graph_history]
            connectivity = [h['global_connectivity'] for h in self.system.graph_history]
            entropy = [h['system_entropy'] / 3.0 for h in self.system.graph_history]  # Normalize
            
            self.ax_scales.plot(generations, connectivity, label='Connectivity', color='#00ff88', linewidth=2)
            self.ax_scales.plot(generations, entropy, label='Entropy', color='#ff6b6b', linewidth=2)
            
            self.ax_scales.set_ylabel('Metric Value', color='white')
            self.ax_scales.legend(fontsize=8, facecolor='#333333', edgecolor='white')
            self.ax_scales.tick_params(colors='white')
        
        self.ax_scales.set_title('System Metrics', color='white', fontsize=10)
    
    def _draw_multiscale_timeline(self):
        """Draw comprehensive timeline."""
        if len(self.system.ca_history) > 1:
            # CA timeline
            ca_gens = [h['generation'] for h in self.system.ca_history]
            ca_density = [h['density'] * 100 for h in self.system.ca_history]  # Percentage
            
            self.ax_timeline.plot(ca_gens, ca_density, label='CA Density %', color='#00ffff', linewidth=1, alpha=0.7)
        
        if len(self.system.graph_history) > 1:
            # Graph timeline  
            graph_gens = [h['generation'] for h in self.system.graph_history]
            num_patterns = [h['num_patterns'] for h in self.system.graph_history]
            num_edges = [h['num_edges'] for h in self.system.graph_history]
            
            self.ax_timeline.plot(graph_gens, num_patterns, label='Patterns', color='#ff6b6b', linewidth=2, marker='o', markersize=2)
            self.ax_timeline.plot(graph_gens, num_edges, label='Connections', color='#4ecdc4', linewidth=2, marker='s', markersize=2)
        
        if len(self.system.meta_history) > 1:
            # Meta timeline
            meta_gens = [h['generation'] for h in self.system.meta_history]
            meta_count = [h['num_meta_patterns'] * 5 for h in self.system.meta_history]  # Scale up for visibility
            
            self.ax_timeline.plot(meta_gens, meta_count, label='Meta-Patterns ×5', color='#ff9ff3', linewidth=3, marker='^', markersize=3)
        
        self.ax_timeline.set_xlabel('Generation', color='white')
        self.ax_timeline.set_ylabel('Count / %', color='white')
        self.ax_timeline.set_title('Multi-Scale Timeline', color='white', fontsize=12)
        self.ax_timeline.legend(loc='upper left', fontsize=9, facecolor='#333333', edgecolor='white')
        self.ax_timeline.grid(True, color='#333333', alpha=0.3)
        self.ax_timeline.tick_params(colors='white')
    
    def run_demo(self, max_generations: int = 300, steps_per_second: float = 6):
        """Run multi-scale demo."""
        generation_count = 0
        
        def animate(frame):
            nonlocal generation_count
            if generation_count < max_generations:
                # Multi-scale evolution
                self.system.step_ca()
                self.system.detect_patterns_multiscale()
                self.system.evolve_graph_structure()
                self.system.analyze_meta_patterns()
                
                influenced = self.system.apply_global_influence()
                if influenced:
                    print(f"Gen {self.system.generation}: Global {self.system.memory.global_phase} influence applied")
                
                # Update visualization
                self.update_visualization()
                
                # Progress logging
                if generation_count % 30 == 0:
                    stats = self.system.get_multiscale_stats()
                    print(f"Gen {stats['generation']}: {stats['num_patterns']} patterns, "
                          f"{stats['num_meta_patterns']} meta-patterns, Phase: {stats['global_phase']}")
                
                generation_count += 1
            return []
        
        interval = int(1000 / steps_per_second)
        ani = animation.FuncAnimation(self.fig, animate, interval=interval, blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
        return ani

def main():
    """Multi-scale emergent graph demonstration."""
    print("Multi-Scale Emergent Graph System")
    print("=" * 40)
    print("Features:")
    print("- CA evolves at fast timescale")
    print("- Graph structure at medium timescale") 
    print("- Meta-patterns at slow timescale")
    print("- Global phases at slowest timescale")
    print("- Memory and historical influences")
    print()
    
    system = MultiScaleEmergentSystem(ca_size=(90, 90))
    system.initialize_ca(density=0.3, seed=42)
    
    viz = MultiScaleVisualizer(system)
    
    try:
        print("Starting multi-scale demonstration...")
        viz.run_demo(max_generations=250, steps_per_second=5)
    except KeyboardInterrupt:
        print("\\nDemo stopped by user")
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    main()