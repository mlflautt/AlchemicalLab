"""
NumPy-based Semantic CA + Mobile Agents

Working implementation without JAX dependencies for immediate visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Agent:
    """Mobile agent with game-like properties."""
    id: int
    x: float
    y: float
    species: int
    energy: float
    resources: np.ndarray
    age: int
    behavior_type: str
    memory: Dict[str, Any]
    color: np.ndarray

class NumpySemanticCA:
    """NumPy-based semantic cellular automaton."""
    
    def __init__(self, size=(50, 50), seed=42):
        np.random.seed(seed)
        self.size = size
        h, w = size
        
        # Initialize grid state
        self.alive = np.random.random((h, w)) < 0.3
        self.species = np.random.randint(0, 5, (h, w))
        self.energy = np.random.uniform(0, 1, (h, w))
        self.resources = np.random.uniform(0, 1, (h, w))
        self.color = np.random.uniform(0, 1, (h, w, 3))
        
        print(f"Initialized {size} NumPy CA")
    
    def get_neighbors(self, grid, i, j):
        """Get 3x3 neighborhood with wrapping."""
        h, w = grid.shape
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = (i + di) % h, (j + dj) % w
                neighbors.append(grid[ni, nj])
        return np.array(neighbors)
    
    def step(self):
        """Execute one simulation step."""
        h, w = self.size
        new_alive = np.zeros_like(self.alive)
        new_energy = np.zeros_like(self.energy)
        
        for i in range(h):
            for j in range(w):
                # Count alive neighbors
                alive_neighbors = self.get_neighbors(self.alive, i, j)
                neighbor_count = np.sum(alive_neighbors) - self.alive[i, j]
                
                current_alive = self.alive[i, j]
                current_energy = self.energy[i, j]
                
                # Game of Life + Energy rules
                if current_alive:
                    # Survival: 2-3 neighbors and energy > 0.2
                    survives = (neighbor_count >= 2) and (neighbor_count <= 3) and (current_energy > 0.2)
                    new_energy[i, j] = max(0.0, current_energy - 0.01) if survives else 0.0
                else:
                    # Birth: exactly 3 neighbors
                    survives = neighbor_count == 3
                    new_energy[i, j] = 0.5 if survives else 0.0
                
                new_alive[i, j] = survives
                
                # Update color based on species and energy
                if survives:
                    species_color = self.species[i, j] / 5.0
                    energy_factor = new_energy[i, j]
                    self.color[i, j] = [species_color, energy_factor, 0.5]
                else:
                    self.color[i, j] = [0.0, 0.0, 0.0]
        
        self.alive = new_alive
        self.energy = new_energy
        
        # Decay resources
        self.resources = np.maximum(0.0, self.resources - 0.001)
        
        return self.get_stats()
    
    def get_stats(self):
        """Get simulation statistics."""
        return {
            'alive_count': int(np.sum(self.alive)),
            'avg_energy': float(np.mean(self.energy)),
            'species_diversity': len(np.unique(self.species)),
            'total_resources': float(np.sum(self.resources))
        }
    
    def get_visualization(self):
        """Get visualization data."""
        return {
            'alive': self.alive,
            'species': self.species,
            'energy': self.energy,
            'color': self.color,
            'resources': self.resources
        }

class NumpyAgentSystem:
    """NumPy-based mobile agent system."""
    
    def __init__(self, world: NumpySemanticCA, max_agents=15, seed=42):
        np.random.seed(seed + 1)  # Different seed for agents
        self.world = world
        self.max_agents = max_agents
        self.agents: List[Agent] = []
        self.next_id = 0
        
        self.spawn_initial_agents(10)
    
    def spawn_initial_agents(self, count: int):
        """Spawn initial population of agents."""
        h, w = self.world.size
        
        behaviors = ['explorer', 'trader', 'builder', 'predator']
        behavior_colors = {
            'explorer': np.array([0.0, 1.0, 0.0]),
            'trader': np.array([1.0, 1.0, 0.0]),
            'builder': np.array([0.0, 0.0, 1.0]),
            'predator': np.array([1.0, 0.0, 0.0])
        }
        
        for i in range(count):
            x = np.random.uniform(0, w-1)
            y = np.random.uniform(0, h-1)
            species = np.random.randint(0, 5)
            energy = np.random.uniform(0.5, 1.0)
            resources = np.random.uniform(0, 0.5, 3)
            
            behavior = behaviors[species % len(behaviors)]
            
            agent = Agent(
                id=self.next_id,
                x=x, y=y,
                species=species,
                energy=energy,
                resources=resources,
                age=0,
                behavior_type=behavior,
                memory={'target_x': x, 'target_y': y, 'state': 'wandering'},
                color=behavior_colors[behavior]
            )
            
            self.agents.append(agent)
            self.next_id += 1
    
    def step(self):
        """Update all agents."""
        # Remove dead agents
        self.agents = [a for a in self.agents if a.energy > 0.05]
        
        # Update surviving agents
        for agent in self.agents:
            self.update_agent(agent)
        
        # Spawn new agents if population low
        if len(self.agents) < self.max_agents // 2:
            self.spawn_initial_agents(2)
    
    def update_agent(self, agent: Agent):
        """Update single agent."""
        h, w = self.world.size
        
        # Simple movement behavior
        if agent.behavior_type == 'explorer':
            # Random walk
            agent.x += np.random.uniform(-0.5, 0.5)
            agent.y += np.random.uniform(-0.5, 0.5)
        elif agent.behavior_type == 'predator':
            # Move toward center (simplified hunting)
            agent.x += 0.1 * (w/2 - agent.x) / w
            agent.y += 0.1 * (h/2 - agent.y) / h
        
        # Wrap around world
        agent.x = agent.x % w
        agent.y = agent.y % h
        
        # Age and energy decay
        agent.age += 1
        agent.energy = max(0.0, agent.energy - 0.005)
        
        # Interact with world cell
        i, j = int(agent.y) % h, int(agent.x) % w
        if self.world.alive[i, j] and self.world.resources[i, j] > 0.1:
            # Gain resources
            resource_gain = min(0.01, self.world.resources[i, j] * 0.1)
            agent.resources[0] += resource_gain
            self.world.resources[i, j] -= resource_gain
    
    def get_stats(self):
        """Get agent statistics."""
        if not self.agents:
            return {'total_agents': 0}
        
        behavior_counts = {}
        for agent in self.agents:
            behavior_counts[agent.behavior_type] = behavior_counts.get(agent.behavior_type, 0) + 1
        
        return {
            'total_agents': len(self.agents),
            'avg_energy': np.mean([a.energy for a in self.agents]),
            'avg_age': np.mean([a.age for a in self.agents]),
            'behavior_counts': behavior_counts
        }
    
    def get_visualization_data(self):
        """Get visualization data."""
        if not self.agents:
            return {'positions': [], 'colors': [], 'behaviors': []}
        
        return {
            'positions': [(a.x, a.y) for a in self.agents],
            'colors': [a.color for a in self.agents],
            'behaviors': [a.behavior_type for a in self.agents],
            'energies': [a.energy for a in self.agents]
        }

class NumpyVisualizer:
    """Matplotlib visualizer for NumPy simulation."""
    
    def __init__(self, world_size=(40, 40), max_agents=12):
        self.world = NumpySemanticCA(size=world_size, seed=42)
        self.agents = NumpyAgentSystem(self.world, max_agents=max_agents, seed=123)
        
        # Setup plots
        self.fig, (self.ax_world, self.ax_stats) = plt.subplots(1, 2, figsize=(12, 6))
        
        # World display
        self.ax_world.set_title('Semantic CA + Mobile Agents (NumPy)')
        self.ax_world.set_xlim(0, world_size[1])
        self.ax_world.set_ylim(0, world_size[0])
        self.ax_world.set_aspect('equal')
        
        # Stats
        self.ax_stats.set_title('System Statistics')
        
        # Data tracking
        self.steps = []
        self.world_alive_counts = []
        self.agent_counts = []
        self.step_count = 0
        
        self.world_image = None
        self.agent_circles = []
    
    def update_display(self):
        """Update visualization."""
        # Update world
        viz_data = self.world.get_visualization()
        display_grid = viz_data['color']
        
        if self.world_image is None:
            self.world_image = self.ax_world.imshow(display_grid, origin='lower', 
                                                   extent=[0, self.world.size[1], 0, self.world.size[0]])
        else:
            self.world_image.set_data(display_grid)
        
        # Clear previous agent circles
        for circle in self.agent_circles:
            circle.remove()
        self.agent_circles = []
        
        # Update agents
        agent_viz = self.agents.get_visualization_data()
        if agent_viz['positions']:
            for i, (x, y) in enumerate(agent_viz['positions']):
                behavior = agent_viz['behaviors'][i]
                energy = agent_viz['energies'][i]
                
                colors = {
                    'explorer': 'lime', 'trader': 'yellow',
                    'builder': 'cyan', 'predator': 'red'
                }
                color = colors.get(behavior, 'white')
                radius = 0.3 + 0.4 * energy
                
                circle = Circle((x, y), radius, color=color, alpha=0.8, zorder=10)
                self.ax_world.add_patch(circle)
                self.agent_circles.append(circle)
        
        # Update stats
        self.ax_stats.clear()
        if len(self.steps) > 1:
            self.ax_stats.plot(self.steps, self.world_alive_counts, 'g-', label='World Alive')
            self.ax_stats.plot(self.steps, self.agent_counts, 'r-', label='Agents')
            self.ax_stats.legend()
        
        self.ax_stats.set_title(f'Step {self.step_count}')
        
        # Add stats text
        if self.agents.agents:
            stats = self.agents.get_stats()
            behavior_counts = stats.get('behavior_counts', {})
            behavior_text = ', '.join([f"{k}:{v}" for k, v in behavior_counts.items()])
            
            text = f"Alive Cells: {self.world_alive_counts[-1] if self.world_alive_counts else 0}\\n"
            text += f"Agents: {self.agent_counts[-1] if self.agent_counts else 0}\\n"
            text += f"Behaviors: {behavior_text}"
            
            self.ax_stats.text(0.02, 0.98, text, transform=self.ax_stats.transAxes,
                              fontsize=8, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def update_simulation(self):
        """Run one simulation step and update display."""
        world_stats = self.world.step()
        self.agents.step()
        agent_stats = self.agents.get_stats()
        
        # Track data
        self.steps.append(self.step_count)
        self.world_alive_counts.append(world_stats['alive_count'])
        self.agent_counts.append(agent_stats.get('total_agents', 0))
        
        self.update_display()
        self.step_count += 1
        
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}: "
                  f"Cells={world_stats['alive_count']}, "
                  f"Agents={agent_stats.get('total_agents', 0)}")
    
    def run_animation(self, steps_per_second=3):
        """Run animated visualization."""
        def animate(frame):
            self.update_simulation()
            return []
        
        interval = int(1000 / steps_per_second)
        ani = animation.FuncAnimation(self.fig, animate, interval=interval, blit=False)
        
        plt.tight_layout()
        plt.show()
        return ani

def run_numpy_simulation():
    """Run NumPy-based simulation."""
    print("AlchemicalLab NumPy Simulation")
    print("=" * 40)
    
    try:
        viz = NumpyVisualizer(world_size=(30, 30), max_agents=10)
        print("Starting visualization...")
        ani = viz.run_animation(steps_per_second=4)
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Running headless simulation...")
        
        # Headless fallback
        world = NumpySemanticCA(size=(20, 20), seed=42)
        agents = NumpyAgentSystem(world, max_agents=8, seed=123)
        
        for step in range(15):
            world_stats = world.step()
            agents.step()
            agent_stats = agents.get_stats()
            
            print(f"Step {step:2d}: "
                  f"Cells={world_stats['alive_count']:3d}, "
                  f"Agents={agent_stats.get('total_agents', 0):2d}, "
                  f"Energy={world_stats['avg_energy']:.3f}")

if __name__ == "__main__":
    run_numpy_simulation()