"""
Mobile Agents System for AlchemicalLab
=====================================

Adds mobile entities that can move across the Semantic CA world grid.
These agents can interact with cells, other agents, and the environment.

Game-like entities: Creatures, traders, explorers, civilizations
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from test_semantic_ca import SimpleSemanticCA

@dataclass
class Agent:
    """Mobile agent with game-like properties."""
    id: int
    x: float  # Position (can be fractional)
    y: float
    species: int
    energy: float
    resources: jnp.ndarray  # Carried resources
    age: int
    behavior_type: str  # 'explorer', 'trader', 'builder', 'predator'
    memory: Dict[str, Any]  # Agent memory for decisions
    color: jnp.ndarray  # Visual representation

class AgentSystem:
    """System for managing mobile agents in the semantic CA world."""
    
    def __init__(self, world: SimpleSemanticCA, max_agents: int = 100, seed: int = 42):
        self.world = world
        self.max_agents = max_agents
        self.key = random.PRNGKey(seed)
        self.agents: List[Agent] = []
        self.next_id = 0
        
        # Initialize starting agents
        self.spawn_initial_agents(10)
    
    def spawn_initial_agents(self, count: int):
        """Spawn initial population of agents."""
        h, w = self.world.size
        
        for i in range(count):
            self.key, subkey = random.split(self.key)
            keys = random.split(subkey, 8)
            
            # Random position
            x = random.uniform(keys[0], minval=0, maxval=w-1)
            y = random.uniform(keys[1], minval=0, maxval=h-1)
            
            # Random properties
            species = random.randint(keys[2], (), 0, 5)
            energy = random.uniform(keys[3], (), minval=0.5, maxval=1.0)
            resources = random.uniform(keys[4], (3,), minval=0.0, maxval=0.5)
            
            # Behavior type based on species
            behaviors = ['explorer', 'trader', 'builder', 'predator']
            behavior_idx = int(species) % len(behaviors)
            behavior = behaviors[behavior_idx]
            
            # Color based on behavior
            behavior_colors = {
                'explorer': jnp.array([0.0, 1.0, 0.0]),  # Green
                'trader': jnp.array([1.0, 1.0, 0.0]),    # Yellow
                'builder': jnp.array([0.0, 0.0, 1.0]),   # Blue
                'predator': jnp.array([1.0, 0.0, 0.0])   # Red
            }
            
            agent = Agent(
                id=self.next_id,
                x=float(x),
                y=float(y),
                species=int(species),
                energy=float(energy),
                resources=resources,
                age=0,
                behavior_type=behavior,
                memory={'target_x': float(x), 'target_y': float(y), 'state': 'wandering'},
                color=behavior_colors[behavior]
            )
            
            self.agents.append(agent)
            self.next_id += 1
    
    def get_cell_at_position(self, x: float, y: float) -> Dict[str, Any]:
        """Get the cell properties at a given position."""
        h, w = self.world.size
        i, j = int(y) % h, int(x) % w
        
        cell = {}
        for prop, grid in self.world.state.items():
            if len(grid.shape) == 2:  # 2D grid
                cell[prop] = grid[i, j]
            else:  # 3D grid (like color, resources)
                cell[prop] = grid[i, j]
        return cell
    
    def find_nearby_agents(self, agent: Agent, radius: float = 3.0) -> List[Agent]:
        """Find other agents within radius."""
        nearby = []
        for other in self.agents:
            if other.id != agent.id:
                dx = agent.x - other.x
                dy = agent.y - other.y
                distance = jnp.sqrt(dx*dx + dy*dy)
                if distance <= radius:
                    nearby.append(other)
        return nearby
    
    def explorer_behavior(self, agent: Agent) -> Tuple[float, float]:
        """Explorer agent behavior: seeks unexplored areas."""
        h, w = self.world.size
        
        # If no target or reached target, pick new random target
        if agent.memory['state'] == 'wandering':
            self.key, subkey = random.split(self.key)
            target_x = random.uniform(subkey, minval=0, maxval=w-1)
            
            self.key, subkey = random.split(self.key)
            target_y = random.uniform(subkey, minval=0, maxval=h-1)
            
            agent.memory['target_x'] = float(target_x)
            agent.memory['target_y'] = float(target_y)
            agent.memory['state'] = 'moving'
        
        # Move toward target
        dx = agent.memory['target_x'] - agent.x
        dy = agent.memory['target_y'] - agent.y
        distance = jnp.sqrt(dx*dx + dy*dy)
        
        if distance < 1.0:  # Reached target
            agent.memory['state'] = 'wandering'
            return 0.0, 0.0
        
        # Normalize and apply speed
        speed = 0.5
        return speed * dx / distance, speed * dy / distance
    
    def trader_behavior(self, agent: Agent) -> Tuple[float, float]:
        """Trader agent behavior: seeks other agents to trade with."""
        nearby = self.find_nearby_agents(agent, radius=5.0)
        
        if nearby:
            # Move toward nearest agent
            target = nearby[0]
            dx = target.x - agent.x
            dy = target.y - agent.y
            distance = jnp.sqrt(dx*dx + dy*dy)
            
            if distance > 1.0:  # Move closer
                speed = 0.3
                return speed * dx / distance, speed * dy / distance
            else:  # Close enough to trade
                # Simple resource exchange
                if agent.resources[0] > target.resources[0]:
                    # Transfer some resources
                    transfer = min(0.1, agent.resources[0] * 0.1)
                    agent.resources = agent.resources.at[0].add(-transfer)
                    target.resources = target.resources.at[0].add(transfer)
                return 0.0, 0.0
        else:
            # No one nearby, explore
            return self.explorer_behavior(agent)
    
    def builder_behavior(self, agent: Agent) -> Tuple[float, float]:
        """Builder agent behavior: modifies the environment."""
        cell = self.get_cell_at_position(agent.x, agent.y)
        
        # If on a dead cell with energy, try to revive it
        if not cell['alive'] and agent.energy > 0.5:
            i, j = int(agent.y) % self.world.size[0], int(agent.x) % self.world.size[1]
            
            # Transfer energy to cell
            self.world.state['alive'] = self.world.state['alive'].at[i, j].set(True)
            self.world.state['energy'] = self.world.state['energy'].at[i, j].add(0.3)
            agent.energy -= 0.1
        
        # Move slowly in search of dead cells
        return self.explorer_behavior(agent)
    
    def predator_behavior(self, agent: Agent) -> Tuple[float, float]:
        """Predator agent behavior: hunts other agents."""
        nearby = self.find_nearby_agents(agent, radius=8.0)
        
        # Filter for weaker agents (less energy)
        prey = [a for a in nearby if a.energy < agent.energy * 0.8]
        
        if prey:
            target = min(prey, key=lambda a: (agent.x - a.x)**2 + (agent.y - a.y)**2)
            dx = target.x - agent.x
            dy = target.y - agent.y
            distance = jnp.sqrt(dx*dx + dy*dy)
            
            if distance < 1.0:  # Close enough to attack
                # Transfer energy (simplified predation)
                energy_transfer = min(0.2, target.energy * 0.5)
                agent.energy = min(1.0, agent.energy + energy_transfer)
                target.energy = max(0.0, target.energy - energy_transfer)
                return 0.0, 0.0
            else:
                # Chase prey
                speed = 0.7
                return speed * dx / distance, speed * dy / distance
        else:
            # No prey, wander
            return self.explorer_behavior(agent)
    
    def update_agent(self, agent: Agent):
        """Update a single agent's behavior and position."""
        h, w = self.world.size
        
        # Get movement from behavior
        behaviors = {
            'explorer': self.explorer_behavior,
            'trader': self.trader_behavior,
            'builder': self.builder_behavior,
            'predator': self.predator_behavior
        }
        
        dx, dy = behaviors[agent.behavior_type](agent)
        
        # Update position with world wrapping
        agent.x = (agent.x + dx) % w
        agent.y = (agent.y + dy) % h
        
        # Age and energy decay
        agent.age += 1
        agent.energy = max(0.0, agent.energy - 0.005)  # Slow energy decay
        
        # Interact with current cell
        cell = self.get_cell_at_position(agent.x, agent.y)
        
        # Gain resources from alive cells
        if cell['alive'] and cell['resources'] > 0.1:
            resource_gain = min(0.01, float(cell['resources']) * 0.1)
            agent.resources = agent.resources.at[0].add(resource_gain)
            
            # Update world cell
            i, j = int(agent.y) % h, int(agent.x) % w
            self.world.state['resources'] = self.world.state['resources'].at[i, j].add(-resource_gain)
        
        # Convert resources to energy when needed
        if agent.energy < 0.3 and agent.resources[0] > 0.1:
            energy_gain = min(0.2, float(agent.resources[0]))
            agent.energy = min(1.0, agent.energy + energy_gain)
            agent.resources = agent.resources.at[0].add(-energy_gain)
    
    def step(self):
        """Update all agents for one simulation step."""
        # Remove dead agents
        self.agents = [a for a in self.agents if a.energy > 0.05]
        
        # Update surviving agents
        for agent in self.agents:
            self.update_agent(agent)
        
        # Spawn new agents if population is low
        if len(self.agents) < self.max_agents // 2:
            self.spawn_initial_agents(2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent population statistics."""
        if not self.agents:
            return {'total_agents': 0}
        
        behavior_counts = {}
        for agent in self.agents:
            behavior_counts[agent.behavior_type] = behavior_counts.get(agent.behavior_type, 0) + 1
        
        return {
            'total_agents': len(self.agents),
            'avg_energy': float(np.mean([a.energy for a in self.agents])),
            'avg_age': float(np.mean([a.age for a in self.agents])),
            'behavior_counts': behavior_counts,
            'total_agent_resources': float(np.sum([np.sum(a.resources) for a in self.agents]))
        }
    
    def get_visualization_data(self):
        """Get agent positions and properties for visualization."""
        if not self.agents:
            return {'positions': [], 'colors': [], 'behaviors': []}
        
        positions = [(a.x, a.y) for a in self.agents]
        colors = [np.array(a.color) for a in self.agents]
        behaviors = [a.behavior_type for a in self.agents]
        
        return {
            'positions': positions,
            'colors': colors,
            'behaviors': behaviors,
            'energies': [a.energy for a in self.agents],
            'ages': [a.age for a in self.agents]
        }

# ============================================================================
# Combined World + Agents Simulation
# ============================================================================

def run_world_with_agents():
    """Run combined CA world + mobile agents simulation."""
    print("🌍 AlchemicalLab: World + Agents Simulation")
    print("=" * 60)
    
    # Create world and agents
    world = SimpleSemanticCA(size=(40, 40), seed=42)
    agents = AgentSystem(world, max_agents=20, seed=123)
    
    print("Initial state:")
    world_stats = world.get_stats()
    agent_stats = agents.get_stats()
    
    print("  World:")
    for key, value in world_stats.items():
        print(f"    {key}: {value}")
    
    print("  Agents:")
    for key, value in agent_stats.items():
        print(f"    {key}: {value}")
    
    print("\\nRunning combined simulation...")
    
    for step in range(30):
        # Update world CA
        world.step()
        
        # Update agents
        agents.step()
        
        if step % 5 == 0:
            w_stats = world.get_stats()
            a_stats = agents.get_stats()
            
            print(f"Step {step:2d}: "
                  f"Cells={w_stats['alive_count']:3d}, "
                  f"Agents={a_stats['total_agents']:2d}, "
                  f"World_Energy={w_stats['avg_energy']:.3f}, "
                  f"Agent_Energy={a_stats.get('avg_energy', 0):.3f}")
            
            if 'behavior_counts' in a_stats:
                behaviors = ", ".join([f"{k}:{v}" for k, v in a_stats['behavior_counts'].items()])
                print(f"        Behaviors: {behaviors}")
    
    print("\\n🎮 Emergent agent-world ecosystem complete!")
    
    # Show final agent positions
    viz_data = agents.get_visualization_data()
    print(f"\\nFinal agent distribution:")
    print(f"  Total agents: {len(viz_data['positions'])}")
    if viz_data['positions']:
        print(f"  Positions (first 5): {viz_data['positions'][:5]}")
        print(f"  Behaviors: {set(viz_data['behaviors'])}")
    
    return world, agents

if __name__ == "__main__":
    world, agents = run_world_with_agents()