"""
Simple Matplotlib Visualization for Semantic CA + Mobile Agents

Shows:
- CA grid state (alive/dead cells colored by species/energy)
- Mobile agent positions and types
- Real-time statistics

Nothing fancy, just functional visualization to see what's happening.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle
import time

from test_semantic_ca import SimpleSemanticCA
from mobile_agents import AgentSystem

class CAVisualizer:
    """Simple matplotlib-based visualizer for CA + agents."""
    
    def __init__(self, world_size=(50, 50), max_agents=15):
        self.world_size = world_size
        self.max_agents = max_agents
        
        # Create world and agents
        self.world = SimpleSemanticCA(size=world_size, seed=42)
        self.agents = AgentSystem(self.world, max_agents=max_agents, seed=123)
        
        # Setup matplotlib figure
        self.fig, (self.ax_world, self.ax_stats) = plt.subplots(1, 2, figsize=(12, 6))
        
        # World display
        self.ax_world.set_title('Semantic CA + Mobile Agents')
        self.ax_world.set_xlim(0, world_size[1])
        self.ax_world.set_ylim(0, world_size[0])
        self.ax_world.set_aspect('equal')
        
        # Stats display
        self.ax_stats.set_title('System Statistics')
        self.ax_stats.set_xlim(0, 100)  # Will adjust as needed
        self.ax_stats.set_ylim(0, 1)
        
        # Data storage for plots
        self.steps = []
        self.world_alive_counts = []
        self.world_energies = []
        self.agent_counts = []
        self.agent_energies = []
        
        # Initialize world display
        self.world_image = None
        self.agent_circles = []
        
        self.step_count = 0
        
    def update_world_display(self):
        """Update the world grid visualization."""
        # Get world state
        viz_data = self.world.get_visualization()
        
        # Create combined visualization
        # Base: alive/dead cells
        display_grid = np.zeros((*self.world_size, 3))
        
        # Color alive cells by species (simplified coloring)
        alive_mask = viz_data['alive']
        species_grid = viz_data['species']
        energy_grid = viz_data['energy']
        
        # Color scheme: species determines hue, energy determines brightness
        for i in range(self.world_size[0]):
            for j in range(self.world_size[1]):
                if alive_mask[i, j]:
                    species = species_grid[i, j]
                    energy = energy_grid[i, j]
                    
                    # Simple species coloring
                    if species == 0:
                        display_grid[i, j] = [energy, 0, 0]  # Red spectrum
                    elif species == 1:
                        display_grid[i, j] = [0, energy, 0]  # Green spectrum
                    elif species == 2:
                        display_grid[i, j] = [0, 0, energy]  # Blue spectrum
                    elif species == 3:
                        display_grid[i, j] = [energy, energy, 0]  # Yellow spectrum
                    else:
                        display_grid[i, j] = [energy, 0, energy]  # Magenta spectrum
                else:
                    display_grid[i, j] = [0.1, 0.1, 0.1]  # Dark gray for dead
        
        # Update or create image
        if self.world_image is None:
            self.world_image = self.ax_world.imshow(display_grid, origin='lower', extent=[0, self.world_size[1], 0, self.world_size[0]])
        else:
            self.world_image.set_data(display_grid)
    
    def update_agent_display(self):
        """Update mobile agent visualization."""
        # Clear previous agent circles
        for circle in self.agent_circles:
            circle.remove()
        self.agent_circles = []
        
        # Get agent data
        agent_viz = self.agents.get_visualization_data()
        
        if agent_viz['positions']:
            for i, (x, y) in enumerate(agent_viz['positions']):
                behavior = agent_viz['behaviors'][i]
                energy = agent_viz['energies'][i]
                
                # Color by behavior type
                colors = {
                    'explorer': 'lime',
                    'trader': 'yellow', 
                    'builder': 'cyan',
                    'predator': 'red'
                }
                
                color = colors.get(behavior, 'white')
                
                # Size by energy
                radius = 0.3 + 0.4 * energy  # 0.3 to 0.7 range
                
                circle = Circle((float(x), float(y)), radius, color=color, alpha=0.8, zorder=10)
                self.ax_world.add_patch(circle)
                self.agent_circles.append(circle)
    
    def update_stats_display(self):
        """Update statistics plots."""
        self.ax_stats.clear()
        
        if len(self.steps) > 1:
            # Plot world statistics
            self.ax_stats.plot(self.steps, [c/max(self.world_alive_counts) for c in self.world_alive_counts], 'g-', label='World Alive (norm)', alpha=0.7)
            self.ax_stats.plot(self.steps, self.world_energies, 'g--', label='World Energy', alpha=0.7)
            
            # Plot agent statistics
            self.ax_stats.plot(self.steps, [c/max(max(self.agent_counts), 1) for c in self.agent_counts], 'r-', label='Agent Count (norm)', alpha=0.7)
            if self.agent_energies:
                self.ax_stats.plot(self.steps, self.agent_energies, 'r--', label='Agent Energy', alpha=0.7)
        
        self.ax_stats.set_xlim(max(0, len(self.steps)-50), len(self.steps)+5)
        self.ax_stats.set_ylim(0, 1.1)
        self.ax_stats.legend(fontsize=8)
        self.ax_stats.set_title(f'Step {self.step_count}')
        
        # Add current stats as text
        if self.agents.agents:
            behavior_counts = self.agents.get_stats()['behavior_counts']
            behavior_text = ', '.join([f"{k}:{v}" for k, v in behavior_counts.items()])
            
            stats_text = f"Alive Cells: {self.world_alive_counts[-1] if self.world_alive_counts else 0}\n"
            stats_text += f"Agents: {self.agent_counts[-1] if self.agent_counts else 0}\n"
            stats_text += f"Behaviors: {behavior_text}"
            
            self.ax_stats.text(0.02, 0.98, stats_text, transform=self.ax_stats.transAxes, 
                             fontsize=8, verticalalignment='top', 
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def update_simulation(self):
        """Run one simulation step and update displays."""
        # Update world and agents
        world_stats = self.world.step()
        self.agents.step()
        agent_stats = self.agents.get_stats()
        
        # Store statistics
        self.steps.append(self.step_count)
        self.world_alive_counts.append(world_stats['alive_count'])
        self.world_energies.append(world_stats['avg_energy'])
        self.agent_counts.append(agent_stats.get('total_agents', 0))
        self.agent_energies.append(agent_stats.get('avg_energy', 0))
        
        # Update visualizations
        self.update_world_display()
        self.update_agent_display()
        self.update_stats_display()
        
        self.step_count += 1
        
        # Print occasional updates to console
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}: Cells={world_stats['alive_count']}, Agents={agent_stats.get('total_agents', 0)}")
    
    def run_interactive(self, steps_per_second=2):
        """Run interactive visualization."""
        print(f"Starting visualization of {self.world_size[0]}x{self.world_size[1]} world with up to {self.max_agents} agents")
        print("Close window to stop simulation")
        
        def animate(frame):
            if plt.get_fignums():  # Check if window is still open
                self.update_simulation()
                return []
        
        # Set up animation
        interval = int(1000 / steps_per_second)  # Convert to milliseconds
        ani = animation.FuncAnimation(self.fig, animate, interval=interval, blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
        
        return ani

def run_visualization(world_size=(40, 40), max_agents=12, steps_per_second=3):
    """Run the CA+Agent visualization."""
    viz = CAVisualizer(world_size=world_size, max_agents=max_agents)
    return viz.run_interactive(steps_per_second=steps_per_second)

if __name__ == "__main__":
    print("AlchemicalLab: Semantic CA + Mobile Agents Visualization")
    print("=" * 55)
    
    # Check if display is available
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Try to use TkAgg backend
        ani = run_visualization()
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Running headless simulation instead...")
        
        # Fallback: run simulation without graphics
        world = SimpleSemanticCA(size=(30, 30), seed=42)
        agents = AgentSystem(world, max_agents=10, seed=123)
        
        for step in range(20):
            world_stats = world.step()
            agents.step()
            agent_stats = agents.get_stats()
            
            print(f"Step {step:2d}: Cells={world_stats['alive_count']:3d}, "
                  f"Agents={agent_stats.get('total_agents', 0):2d}, "
                  f"World_Energy={world_stats['avg_energy']:.3f}")
        
        print("Headless simulation complete.")