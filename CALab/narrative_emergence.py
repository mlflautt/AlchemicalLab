"""
Narrative World Generation using CA-Graph Emergence
CA patterns evolve into narrative elements: characters, locations, events, relationships
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch, ConnectionPatch, Rectangle
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import time

plt.style.use('dark_background')

# Narrative Element Types
NARRATIVE_TYPES = {
    'character': {'color': '#ff6b6b', 'symbol': '♦', 'influence': 8},
    'location': {'color': '#4ecdc4', 'symbol': '■', 'influence': 12},
    'artifact': {'color': '#feca57', 'symbol': '◆', 'influence': 6},
    'event': {'color': '#ff9ff3', 'symbol': '★', 'influence': 10},
    'faction': {'color': '#54a0ff', 'symbol': '▲', 'influence': 15},
    'mystery': {'color': '#5f27cd', 'symbol': '?', 'influence': 4}
}

RELATIONSHIP_TYPES = {
    'alliance': {'color': '#2ed573', 'strength': 0.8, 'stability': 0.9},
    'conflict': {'color': '#ff4757', 'strength': 0.9, 'stability': 0.6},
    'dependency': {'color': '#ff9ff3', 'strength': 0.7, 'stability': 0.8},
    'mystery_connection': {'color': '#5f27cd', 'strength': 0.5, 'stability': 0.4},
    'location_presence': {'color': '#4ecdc4', 'strength': 0.6, 'stability': 0.9},
    'artifact_ownership': {'color': '#feca57', 'strength': 0.8, 'stability': 0.7}
}

@dataclass
class NarrativeElement:
    """A narrative element derived from CA patterns."""
    element_id: int
    element_type: str  # character, location, artifact, event, faction, mystery
    name: str
    center: Tuple[int, int]
    size: int
    birth_generation: int
    last_seen: int
    
    # Narrative attributes
    properties: Dict[str, Any] = field(default_factory=dict)
    backstory: List[str] = field(default_factory=list)
    relationships: Set[int] = field(default_factory=set)
    influence_radius: float = 10.0
    narrative_weight: float = 1.0
    
    # Dynamic attributes
    activity_level: float = 1.0
    stability: float = 0.5
    mystery_level: float = 0.0

@dataclass
class NarrativeRelationship:
    """Relationship between narrative elements."""
    source_id: int
    target_id: int
    relationship_type: str
    strength: float = 1.0
    age: int = 0
    story_events: List[str] = field(default_factory=list)
    
    # Dynamic relationship properties
    tension: float = 0.0
    development_arc: str = "stable"  # stable, growing, declining, volatile

@dataclass
class StoryArc:
    """A developing story arc involving multiple elements."""
    arc_id: int
    arc_type: str  # quest, conflict, romance, mystery, political
    participants: Set[int]
    central_location: Optional[int] = None
    key_artifacts: Set[int] = field(default_factory=set)
    story_beats: List[str] = field(default_factory=list)
    completion: float = 0.0
    tension_level: float = 0.5

class NarrativeNameGenerator:
    """Generates narrative names and descriptions."""
    
    def __init__(self):
        self.character_names = [
            "Aria", "Kael", "Luna", "Zara", "Thorne", "Elysia", "Dante", "Vera",
            "Orion", "Maya", "Casper", "Iris", "Felix", "Nova", "Sage", "Raven"
        ]
        
        self.location_names = [
            "Shadowmere", "Goldspire", "Ironhold", "Mistport", "Thornwall", "Silverbrook",
            "Crimson Keep", "Azure Falls", "Ember Vale", "Frost Gate", "Storm's End"
        ]
        
        self.artifact_names = [
            "Starfall Pendant", "Void Blade", "Whisper Stone", "Phoenix Crown", "Soul Mirror",
            "Storm Chalice", "Shadow Cloak", "Dragon's Heart", "Moonwell Orb", "Eternal Flame"
        ]
        
        self.event_types = [
            "Great Convergence", "Shadow Eclipse", "Royal Wedding", "Uprising", "Discovery",
            "Betrayal", "Siege", "Celebration", "Prophecy", "Catastrophe", "Revelation"
        ]
        
        self.faction_names = [
            "Crimson Order", "Silver Council", "Shadow Guild", "Iron Brotherhood", "Starlight Assembly",
            "Void Covenant", "Phoenix League", "Storm Wardens", "Moonlight Circle", "Dragon's Alliance"
        ]
        
        self.mysteries = [
            "Lost Heir", "Ancient Curse", "Hidden Treasure", "Forbidden Love", "Secret Alliance",
            "Prophecy", "Lost Kingdom", "Sacred Artifact", "Time Rift", "Soul Bond"
        ]
        
    def generate_name(self, element_type: str, properties: Dict = None) -> str:
        """Generate a name for a narrative element."""
        if element_type == 'character':
            return random.choice(self.character_names)
        elif element_type == 'location':
            return random.choice(self.location_names)
        elif element_type == 'artifact':
            return random.choice(self.artifact_names)
        elif element_type == 'event':
            return f"The {random.choice(self.event_types)}"
        elif element_type == 'faction':
            return random.choice(self.faction_names)
        elif element_type == 'mystery':
            return f"Mystery of the {random.choice(self.mysteries)}"
        else:
            return f"Unknown {element_type.title()}"

class WorldBuildingSystem:
    """CA-driven narrative world generation system."""
    
    def __init__(self, world_size: Tuple[int, int] = (120, 120)):
        self.world_size = world_size
        self.ca_grid = np.zeros(world_size, dtype=bool)
        self.generation = 0
        
        # Narrative system
        self.narrative_elements: Dict[int, NarrativeElement] = {}
        self.relationships: Dict[Tuple[int, int], NarrativeRelationship] = {}
        self.story_arcs: Dict[int, StoryArc] = {}
        
        self.element_id_counter = 0
        self.arc_id_counter = 0
        self.name_generator = NarrativeNameGenerator()
        
        # Pattern detection (from previous system)
        from emergent_graphs import CAPatternDetector
        self.detector = CAPatternDetector()
        
        # World generation parameters
        self.element_spawn_threshold = 8  # Min pattern size for narrative element
        self.relationship_threshold = 25.0  # Distance for relationships
        self.arc_formation_threshold = 3  # Min elements for story arc
        
        # Narrative memory and evolution
        self.world_history: List[Dict] = []
        self.active_storylines: Set[int] = set()
        
        # Regional characteristics (influences narrative type spawning)
        self.regions = {
            'political': {'center': (30, 30), 'radius': 25, 'element_bias': ['faction', 'character']},
            'mystical': {'center': (90, 30), 'radius': 20, 'element_bias': ['mystery', 'artifact']},
            'wilderness': {'center': (60, 90), 'radius': 30, 'element_bias': ['location', 'event']},
            'urban': {'center': (30, 90), 'radius': 20, 'element_bias': ['character', 'faction']},
            'ancient': {'center': (90, 90), 'radius': 15, 'element_bias': ['artifact', 'mystery']}
        }
        
    def initialize_world(self, density: float = 0.25, seed: int = 42):
        """Initialize the world with regional seeding."""
        np.random.seed(seed)
        random.seed(seed)
        
        h, w = self.world_size
        
        # Create different regions with varying characteristics
        for region_name, region_data in self.regions.items():
            center_y, center_x = region_data['center']
            radius = region_data['radius']
            
            # Create regional pattern
            y_coords, x_coords = np.ogrid[:h, :w]
            distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            region_mask = distances <= radius
            
            # Add regional characteristics
            region_density = density * random.uniform(0.8, 1.5)
            self.ca_grid[region_mask] |= np.random.random(np.sum(region_mask)) < region_density
        
        # Add some connecting paths between regions
        self._add_connecting_paths()
        
        # Add seed patterns for narrative bootstrap
        self._add_narrative_seeds()
    
    def _add_connecting_paths(self):
        """Add paths connecting different regions."""
        region_centers = [data['center'] for data in self.regions.values()]
        
        for i, (y1, x1) in enumerate(region_centers):
            for y2, x2 in region_centers[i+1:]:
                # Create sparse connecting path
                steps = max(abs(y2-y1), abs(x2-x1))
                for step in range(steps):
                    t = step / steps
                    y = int(y1 + t * (y2 - y1))
                    x = int(x1 + t * (x2 - x1))
                    
                    if 0 <= y < self.world_size[0] and 0 <= x < self.world_size[1]:
                        if random.random() < 0.3:  # Sparse path
                            self.ca_grid[y, x] = True
    
    def _add_narrative_seeds(self):
        """Add specific seed patterns that will become key narrative elements."""
        # Add structured patterns in each region
        for region_name, region_data in self.regions.items():
            center_y, center_x = region_data['center']
            
            # Add a central hub pattern
            hub_pattern = np.array([
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 1, 1],
                [0, 1, 1, 1, 0]
            ], dtype=bool)
            
            y_start = max(0, center_y - 2)
            x_start = max(0, center_x - 2)
            y_end = min(self.world_size[0], y_start + 5)
            x_end = min(self.world_size[1], x_start + 5)
            
            self.ca_grid[y_start:y_end, x_start:x_end] |= hub_pattern[:y_end-y_start, :x_end-x_start]
    
    def step_ca(self):
        """Step the cellular automaton with world-building modifications."""
        from scipy import ndimage
        
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        padded = np.pad(self.ca_grid.astype(int), 1, mode='wrap')
        neighbors = ndimage.convolve(padded, kernel, mode='constant')[1:-1, 1:-1]
        
        # Modified Conway rules with narrative influences
        new_grid = np.zeros_like(self.ca_grid)
        
        # Standard rules
        new_grid |= (~self.ca_grid) & (neighbors == 3)
        new_grid |= self.ca_grid & ((neighbors == 2) | (neighbors == 3))
        
        # Narrative influence on CA evolution
        self._apply_narrative_influences(new_grid, neighbors)
        
        self.ca_grid = new_grid
        self.generation += 1
    
    def _apply_narrative_influences(self, new_grid: np.ndarray, neighbors: np.ndarray):
        """Apply narrative element influences to CA evolution."""
        for element in self.narrative_elements.values():
            y, x = element.center
            radius = int(element.influence_radius)
            
            # Create influence region
            y_min, y_max = max(0, y-radius), min(self.world_size[0], y+radius+1)
            x_min, x_max = max(0, x-radius), min(self.world_size[1], x+radius+1)
            
            # Apply element-type specific influences
            if element.element_type == 'location':
                # Locations stabilize their surroundings
                stability_mask = np.random.random((y_max-y_min, x_max-x_min)) < 0.1
                new_grid[y_min:y_max, x_min:x_max] |= stability_mask
                
            elif element.element_type == 'event':
                # Events create temporary activity
                if element.activity_level > 0.8:
                    activity_mask = np.random.random((y_max-y_min, x_max-x_min)) < 0.2
                    new_grid[y_min:y_max, x_min:x_max] |= activity_mask
                    
            elif element.element_type == 'faction':
                # Factions create structured patterns
                structure_chance = element.narrative_weight * 0.1
                structure_mask = np.random.random((y_max-y_min, x_max-x_min)) < structure_chance
                new_grid[y_min:y_max, x_min:x_max] |= structure_mask
    
    def detect_and_generate_narrative_elements(self):
        """Convert CA patterns into narrative elements."""
        if self.generation % 3 == 0:  # Every 3 generations
            current_patterns = self.detector.detect_patterns(self.ca_grid, self.generation)
            
            for pattern in current_patterns:
                if pattern.size >= self.element_spawn_threshold:
                    # Determine if this should become a narrative element
                    if not self._matches_existing_element(pattern):
                        element = self._create_narrative_element(pattern)
                        if element:
                            self.narrative_elements[element.element_id] = element
    
    def _matches_existing_element(self, pattern) -> bool:
        """Check if pattern matches an existing narrative element."""
        for element in self.narrative_elements.values():
            distance = np.sqrt((pattern.center[0] - element.center[0])**2 + 
                             (pattern.center[1] - element.center[1])**2)
            if distance <= max(8, element.size):
                # Update existing element
                element.last_seen = self.generation
                element.center = pattern.center
                element.activity_level = min(1.0, element.activity_level + 0.1)
                return True
        return False
    
    def _create_narrative_element(self, pattern) -> Optional[NarrativeElement]:
        """Create a narrative element from a CA pattern."""
        # Determine element type based on regional bias and pattern characteristics
        element_type = self._determine_element_type(pattern)
        
        if element_type:
            element_id = self.element_id_counter
            self.element_id_counter += 1
            
            name = self.name_generator.generate_name(element_type)
            
            # Generate properties based on type and pattern
            properties = self._generate_element_properties(element_type, pattern)
            
            element = NarrativeElement(
                element_id=element_id,
                element_type=element_type,
                name=name,
                center=pattern.center,
                size=pattern.size,
                birth_generation=self.generation,
                last_seen=self.generation,
                properties=properties,
                influence_radius=NARRATIVE_TYPES[element_type]['influence'],
                narrative_weight=random.uniform(0.5, 1.5)
            )
            
            # Generate initial backstory
            element.backstory = self._generate_backstory(element)
            
            return element
        return None
    
    def _determine_element_type(self, pattern) -> Optional[str]:
        """Determine what type of narrative element a pattern should become."""
        y, x = pattern.center
        
        # Check regional bias
        for region_name, region_data in self.regions.items():
            center_y, center_x = region_data['center']
            distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            
            if distance <= region_data['radius']:
                # Bias towards region-specific elements
                biased_types = region_data['element_bias']
                if random.random() < 0.7:  # 70% chance to follow regional bias
                    return random.choice(biased_types)
        
        # Pattern-based determination
        if pattern.size <= 4:
            return random.choice(['character', 'artifact'])
        elif pattern.size <= 8:
            return random.choice(['location', 'event'])
        elif pattern.size <= 15:
            return random.choice(['faction', 'location'])
        else:
            return random.choice(['event', 'mystery'])
    
    def _generate_element_properties(self, element_type: str, pattern) -> Dict[str, Any]:
        """Generate properties specific to element type."""
        properties = {}
        
        if element_type == 'character':
            properties.update({
                'class': random.choice(['Warrior', 'Mage', 'Rogue', 'Noble', 'Merchant', 'Scholar']),
                'motivation': random.choice(['Power', 'Knowledge', 'Revenge', 'Love', 'Honor', 'Wealth']),
                'flaw': random.choice(['Pride', 'Greed', 'Fear', 'Anger', 'Jealousy', 'Naivety'])
            })
            
        elif element_type == 'location':
            properties.update({
                'type': random.choice(['City', 'Castle', 'Temple', 'Forest', 'Mountain', 'Ruin']),
                'atmosphere': random.choice(['Peaceful', 'Mysterious', 'Dangerous', 'Sacred', 'Bustling']),
                'resources': random.choice(['Minerals', 'Magic', 'Trade', 'Knowledge', 'Military'])
            })
            
        elif element_type == 'faction':
            properties.update({
                'ideology': random.choice(['Order', 'Freedom', 'Knowledge', 'Power', 'Nature', 'Progress']),
                'methods': random.choice(['Diplomatic', 'Militant', 'Secretive', 'Economic', 'Religious']),
                'goal': random.choice(['Conquest', 'Protection', 'Discovery', 'Unification', 'Revolution'])
            })
            
        elif element_type == 'artifact':
            properties.update({
                'power': random.choice(['Enhancement', 'Protection', 'Destruction', 'Divination', 'Transformation']),
                'origin': random.choice(['Ancient', 'Divine', 'Crafted', 'Natural', 'Cursed']),
                'requirement': random.choice(['Bloodline', 'Ritual', 'Worthiness', 'Knowledge', 'Sacrifice'])
            })
            
        elif element_type == 'event':
            properties.update({
                'scope': random.choice(['Local', 'Regional', 'Kingdom', 'Continental', 'Cosmic']),
                'impact': random.choice(['Political', 'Social', 'Economic', 'Magical', 'Natural']),
                'duration': random.choice(['Instant', 'Days', 'Months', 'Years', 'Eternal'])
            })
            
        elif element_type == 'mystery':
            properties.update({
                'nature': random.choice(['Historical', 'Supernatural', 'Personal', 'Cosmic', 'Technological']),
                'danger_level': random.choice(['Harmless', 'Risky', 'Deadly', 'Catastrophic']),
                'clues_found': 0
            })
        
        return properties
    
    def _generate_backstory(self, element: NarrativeElement) -> List[str]:
        """Generate initial backstory for an element."""
        backstory = []
        
        if element.element_type == 'character':
            backstory.append(f"{element.name} is a {element.properties['class']} driven by {element.properties['motivation'].lower()}.")
            backstory.append(f"Their greatest flaw is their {element.properties['flaw'].lower()}.")
            
        elif element.element_type == 'location':
            backstory.append(f"{element.name} is a {element.properties['type'].lower()} with a {element.properties['atmosphere'].lower()} atmosphere.")
            backstory.append(f"It is known for its {element.properties['resources'].lower()}.")
            
        elif element.element_type == 'faction':
            backstory.append(f"The {element.name} believes in {element.properties['ideology'].lower()} above all else.")
            backstory.append(f"They pursue their goals through {element.properties['methods'].lower()} means.")
            
        # Add more backstory types as needed
        
        return backstory
    
    def evolve_relationships(self):
        """Evolve relationships between narrative elements."""
        if self.generation % 5 == 0:  # Every 5 generations
            self._detect_new_relationships()
            self._evolve_existing_relationships()
            self._detect_story_arcs()
    
    def _detect_new_relationships(self):
        """Detect new relationships forming between elements."""
        element_ids = list(self.narrative_elements.keys())
        
        for i, id1 in enumerate(element_ids):
            for id2 in element_ids[i+1:]:
                if (id1, id2) not in self.relationships and (id2, id1) not in self.relationships:
                    element1 = self.narrative_elements[id1]
                    element2 = self.narrative_elements[id2]
                    
                    distance = np.sqrt((element1.center[0] - element2.center[0])**2 + 
                                     (element1.center[1] - element2.center[1])**2)
                    
                    if distance <= self.relationship_threshold:
                        relationship_type = self._determine_relationship_type(element1, element2)
                        if relationship_type:
                            key = (min(id1, id2), max(id1, id2))
                            self.relationships[key] = NarrativeRelationship(
                                source_id=id1,
                                target_id=id2,
                                relationship_type=relationship_type,
                                strength=random.uniform(0.3, 0.8)
                            )
                            
                            # Add initial story event
                            event_desc = self._generate_relationship_event(element1, element2, relationship_type)
                            self.relationships[key].story_events.append(event_desc)
    
    def _determine_relationship_type(self, element1: NarrativeElement, element2: NarrativeElement) -> Optional[str]:
        """Determine the type of relationship between two elements."""
        type1, type2 = element1.element_type, element2.element_type
        
        # Type-based relationship rules
        if type1 == 'character' and type2 == 'character':
            return random.choice(['alliance', 'conflict', 'dependency'])
        elif type1 == 'character' and type2 == 'location':
            return 'location_presence'
        elif type1 == 'character' and type2 == 'artifact':
            return 'artifact_ownership'
        elif type1 == 'faction' and type2 == 'faction':
            return random.choice(['alliance', 'conflict'])
        elif type1 == 'character' and type2 == 'faction':
            return random.choice(['alliance', 'conflict'])
        elif 'mystery' in [type1, type2]:
            return 'mystery_connection'
        else:
            # Default relationship
            return random.choice(['alliance', 'dependency', 'mystery_connection'])
    
    def _generate_relationship_event(self, element1: NarrativeElement, element2: NarrativeElement, rel_type: str) -> str:
        """Generate a story event describing the relationship formation."""
        if rel_type == 'alliance':
            return f"{element1.name} and {element2.name} formed an alliance in generation {self.generation}."
        elif rel_type == 'conflict':
            return f"{element1.name} came into conflict with {element2.name} in generation {self.generation}."
        elif rel_type == 'location_presence':
            return f"{element1.name} arrived at {element2.name} in generation {self.generation}."
        elif rel_type == 'artifact_ownership':
            return f"{element1.name} acquired {element2.name} in generation {self.generation}."
        elif rel_type == 'mystery_connection':
            return f"{element1.name} became connected to the mystery of {element2.name} in generation {self.generation}."
        else:
            return f"{element1.name} and {element2.name} became connected in generation {self.generation}."
    
    def _evolve_existing_relationships(self):
        """Evolve existing relationships over time."""
        for key, relationship in self.relationships.items():
            relationship.age += 1
            
            # Relationship evolution based on type and age
            if relationship.relationship_type == 'conflict':
                if relationship.age > 10 and random.random() < 0.1:
                    relationship.development_arc = 'declining'
                    relationship.tension = max(0, relationship.tension - 0.1)
                else:
                    relationship.tension = min(1.0, relationship.tension + 0.05)
                    
            elif relationship.relationship_type == 'alliance':
                if relationship.age > 5:
                    relationship.strength = min(1.0, relationship.strength + 0.02)
                    
            elif relationship.relationship_type == 'mystery_connection':
                if random.random() < 0.05:  # Occasional breakthrough
                    breakthrough = f"New clue discovered in generation {self.generation}."
                    relationship.story_events.append(breakthrough)
                    relationship.strength = min(1.0, relationship.strength + 0.1)
    
    def _detect_story_arcs(self):
        """Detect emerging story arcs from relationship patterns."""
        # Find groups of connected elements
        connected_groups = self._find_connected_groups()
        
        for group in connected_groups:
            if len(group) >= self.arc_formation_threshold:
                arc_type = self._determine_arc_type(group)
                
                arc_id = self.arc_id_counter
                self.arc_id_counter += 1
                
                story_arc = StoryArc(
                    arc_id=arc_id,
                    arc_type=arc_type,
                    participants=set(group),
                    completion=0.0,
                    tension_level=random.uniform(0.3, 0.8)
                )
                
                # Generate initial story beats
                story_arc.story_beats = self._generate_story_beats(story_arc, group)
                
                self.story_arcs[arc_id] = story_arc
                self.active_storylines.add(arc_id)
    
    def _find_connected_groups(self) -> List[List[int]]:
        """Find connected groups of narrative elements."""
        # Build adjacency graph
        adjacency = defaultdict(set)
        for (id1, id2) in self.relationships:
            adjacency[id1].add(id2)
            adjacency[id2].add(id1)
        
        # Find connected components
        visited = set()
        groups = []
        
        for element_id in self.narrative_elements:
            if element_id not in visited:
                group = []
                stack = [element_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        group.append(current)
                        stack.extend(adjacency[current] - visited)
                
                if len(group) > 1:
                    groups.append(group)
        
        return groups
    
    def _determine_arc_type(self, group: List[int]) -> str:
        """Determine the type of story arc for a group."""
        element_types = [self.narrative_elements[eid].element_type for eid in group]
        relationship_types = [rel.relationship_type for rel in self.relationships.values() 
                            if rel.source_id in group and rel.target_id in group]
        
        if 'conflict' in relationship_types:
            return 'conflict'
        elif 'mystery' in element_types:
            return 'mystery'
        elif len([t for t in element_types if t == 'character']) >= 2:
            return 'romance' if random.random() < 0.3 else 'quest'
        elif 'faction' in element_types:
            return 'political'
        else:
            return 'quest'
    
    def _generate_story_beats(self, arc: StoryArc, participants: List[int]) -> List[str]:
        """Generate story beats for an arc."""
        beats = []
        
        if arc.arc_type == 'conflict':
            beats.append("Tensions rise between opposing forces.")
            beats.append("The conflict reaches a crucial turning point.")
            beats.append("The final confrontation determines the outcome.")
            
        elif arc.arc_type == 'mystery':
            beats.append("Strange occurrences hint at a deeper mystery.")
            beats.append("Clues are discovered that point to the truth.")
            beats.append("The mystery is finally unraveled.")
            
        elif arc.arc_type == 'quest':
            beats.append("A quest begins with a clear objective.")
            beats.append("Obstacles and challenges test the heroes.")
            beats.append("The quest reaches its climactic conclusion.")
            
        elif arc.arc_type == 'romance':
            beats.append("Two hearts meet under unusual circumstances.")
            beats.append("Obstacles threaten to keep them apart.")
            beats.append("Love finds a way to overcome all barriers.")
            
        elif arc.arc_type == 'political':
            beats.append("Political machinations begin to unfold.")
            beats.append("Alliances shift as power dynamics change.")
            beats.append("A new order emerges from the chaos.")
        
        return beats
    
    def get_world_stats(self) -> Dict:
        """Get comprehensive world statistics."""
        return {
            'generation': self.generation,
            'num_elements': len(self.narrative_elements),
            'num_relationships': len(self.relationships),
            'num_story_arcs': len(self.story_arcs),
            'element_types': {etype: sum(1 for e in self.narrative_elements.values() 
                                       if e.element_type == etype) 
                            for etype in NARRATIVE_TYPES.keys()},
            'relationship_types': {rtype: sum(1 for r in self.relationships.values() 
                                           if r.relationship_type == rtype) 
                                 for rtype in RELATIONSHIP_TYPES.keys()},
            'active_storylines': len(self.active_storylines),
            'world_density': float(np.mean(self.ca_grid))
        }

    def run_headless(self, num_generations: int):
        """Run the simulation without visualization."""
        print(f"Running headless generation for {num_generations} generations...")
        for i in range(num_generations):
            self.step_ca()
            self.detect_and_generate_narrative_elements()
            self.evolve_relationships()
            if i % 25 == 0:
                print(f"  ... generation {i}")
        print("Headless generation complete.")

    def get_world_data(self) -> Dict:
        """Get the full narrative world data as a serializable dictionary."""
        return {
            "stats": self.get_world_stats(),
            "elements": [e.__dict__ for e in self.narrative_elements.values()],
            "relationships": [r.__dict__ for r in self.relationships.values()],
            "story_arcs": [a.__dict__ for a in self.story_arcs.values()]
        }


class NarrativeWorldVisualizer:
    """Visualizer for the narrative world-building system."""
    
    def __init__(self, world_system: WorldBuildingSystem):
        self.world_system = world_system
        
        plt.rcParams.update({
            'figure.facecolor': '#0a0a0a',
            'axes.facecolor': '#1a1a1a',
            'text.color': '#ffffff',
            'axes.labelcolor': '#ffffff',
        })
        
        self.fig = plt.figure(figsize=(20, 14))
        
        # Complex layout for world visualization
        gs = self.fig.add_gridspec(4, 5, hspace=0.4, wspace=0.3)
        
        self.ax_world = self.fig.add_subplot(gs[0:3, 0:3])     # Main world view
        self.ax_network = self.fig.add_subplot(gs[0:2, 3:5])   # Relationship network
        self.ax_elements = self.fig.add_subplot(gs[2, 3])      # Element types
        self.ax_arcs = self.fig.add_subplot(gs[2, 4])          # Story arcs
        self.ax_timeline = self.fig.add_subplot(gs[3, :])      # Timeline
        
        self.fig.patch.set_facecolor('#0a0a0a')
        for ax in [self.ax_world, self.ax_network, self.ax_elements, self.ax_arcs, self.ax_timeline]:
            ax.set_facecolor('#1a1a1a')
        
        # Colors and styling
        self.cmap_ca = plt.matplotlib.colors.ListedColormap(['#0a0a0a', '#333333'])
        self.stats_history = []
        
    def update_visualization(self):
        """Update all visualization components."""
        # Clear all axes
        for ax in [self.ax_world, self.ax_network, self.ax_elements, self.ax_arcs, self.ax_timeline]:
            ax.clear()
            ax.set_facecolor('#1a1a1a')
        
        self._draw_world_map()
        self._draw_relationship_network()
        self._draw_element_distribution()
        self._draw_story_arcs()
        self._draw_timeline()
    
    def _draw_world_map(self):
        """Draw the main world map with narrative elements."""
        # Draw CA background (terrain)
        self.ax_world.imshow(self.world_system.ca_grid, cmap=self.cmap_ca, interpolation='nearest', alpha=0.3)
        
        # Draw regions
        self._draw_regions()
        
        # Draw narrative elements
        for element in self.world_system.narrative_elements.values():
            element_info = NARRATIVE_TYPES[element.element_type]
            color = element_info['color']
            symbol = element_info['symbol']
            
            y, x = element.center
            
            # Element circle (size based on narrative weight)
            radius = 3 + element.narrative_weight * 2
            circle = Circle((x, y), radius=radius, fill=False, color=color, 
                          linewidth=2, alpha=0.8)
            self.ax_world.add_patch(circle)
            
            # Element symbol and name
            self.ax_world.scatter(x, y, c=color, s=150, marker='o', alpha=0.9, edgecolors='white')
            self.ax_world.text(x, y, symbol, ha='center', va='center', 
                             fontsize=8, color='black', weight='bold')
            
            # Element name (if space permits)
            if len(self.world_system.narrative_elements) < 20:
                self.ax_world.text(x, y-radius-2, element.name, ha='center', va='top',
                                 fontsize=7, color=color, weight='bold',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Draw relationships as lines
        for relationship in self.world_system.relationships.values():
            if relationship.source_id in self.world_system.narrative_elements and relationship.target_id in self.world_system.narrative_elements:
                element1 = self.world_system.narrative_elements[relationship.source_id]
                element2 = self.world_system.narrative_elements[relationship.target_id]
                
                rel_info = RELATIONSHIP_TYPES[relationship.relationship_type]
                color = rel_info['color']
                
                x_coords = [element1.center[1], element2.center[1]]
                y_coords = [element1.center[0], element2.center[0]]
                
                alpha = relationship.strength * 0.7
                width = relationship.strength * 2
                
                self.ax_world.plot(x_coords, y_coords, color=color, 
                                 alpha=alpha, linewidth=width, linestyle='-')
        
        self.ax_world.set_title(f'Narrative World Map (Generation {self.world_system.generation})', 
                              color='white', fontsize=14)
        self.ax_world.set_xlim(0, self.world_system.world_size[1])
        self.ax_world.set_ylim(0, self.world_system.world_size[0])
        self.ax_world.invert_yaxis()  # Match image coordinates
    
    def _draw_regions(self):
        """Draw regional boundaries and labels."""
        for region_name, region_data in self.world_system.regions.items():
            center_y, center_x = region_data['center']
            radius = region_data['radius']
            
            # Region boundary
            circle = Circle((center_x, center_y), radius=radius, fill=False, 
                          color='#555555', linewidth=1, linestyle='--', alpha=0.5)
            self.ax_world.add_patch(circle)
            
            # Region label
            self.ax_world.text(center_x, center_y + radius + 3, region_name.title(),
                             ha='center', va='bottom', fontsize=8, color='#888888',
                             style='italic')
    
    def _draw_relationship_network(self):
        """Draw the relationship network graph."""
        if not self.world_system.narrative_elements:
            self.ax_network.text(0.5, 0.5, 'No narrative elements yet', 
                               transform=self.ax_network.transAxes, ha='center', va='center',
                               color='white', fontsize=10)
            return
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for element_id, element in self.world_system.narrative_elements.items():
            G.add_node(element_id, element_type=element.element_type, name=element.name)
        
        # Add edges
        for relationship in self.world_system.relationships.values():
            G.add_edge(relationship.source_id, relationship.target_id, 
                      relationship_type=relationship.relationship_type,
                      strength=relationship.strength)
        
        if len(G.nodes()) > 0:
            try:
                pos = nx.spring_layout(G, k=1, iterations=50)
            except:
                pos = {node: (random.random(), random.random()) for node in G.nodes()}
            
            # Draw nodes
            for node_id in G.nodes():
                if node_id in self.world_system.narrative_elements:
                    element = self.world_system.narrative_elements[node_id]
                    color = NARRATIVE_TYPES[element.element_type]['color']
                    size = 200 + element.narrative_weight * 100
                    
                    self.ax_network.scatter(pos[node_id][0], pos[node_id][1], 
                                          c=color, s=size, alpha=0.8, edgecolors='white', linewidth=1)
                    
                    # Node labels
                    self.ax_network.text(pos[node_id][0], pos[node_id][1], str(node_id),
                                       ha='center', va='center', fontsize=8, 
                                       color='black', weight='bold')
            
            # Draw edges
            for edge in G.edges(data=True):
                node1, node2, data = edge
                if node1 in pos and node2 in pos:
                    rel_type = data.get('relationship_type', 'alliance')
                    color = RELATIONSHIP_TYPES.get(rel_type, RELATIONSHIP_TYPES['alliance'])['color']
                    strength = data.get('strength', 1.0)
                    
                    x_coords = [pos[node1][0], pos[node2][0]]
                    y_coords = [pos[node1][1], pos[node2][1]]
                    
                    self.ax_network.plot(x_coords, y_coords, color=color,
                                       alpha=strength*0.8, linewidth=strength*2)
        
        self.ax_network.set_title('Relationship Network', color='white', fontsize=12)
        self.ax_network.axis('off')
    
    def _draw_element_distribution(self):
        """Draw distribution of narrative element types."""
        stats = self.world_system.get_world_stats()
        element_types = stats['element_types']
        
        if sum(element_types.values()) > 0:
            types = [t for t, c in element_types.items() if c > 0]
            counts = [element_types[t] for t in types]
            colors = [NARRATIVE_TYPES[t]['color'] for t in types]
            
            bars = self.ax_elements.bar(range(len(types)), counts, color=colors, alpha=0.8)
            self.ax_elements.set_xticks(range(len(types)))
            self.ax_elements.set_xticklabels(types, rotation=45, ha='right', fontsize=8)
            self.ax_elements.set_ylabel('Count', color='white', fontsize=9)
            self.ax_elements.tick_params(colors='white')
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                self.ax_elements.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                    str(count), ha='center', va='bottom', fontsize=8, color='white')
        
        self.ax_elements.set_title('Element Types', color='white', fontsize=10)
    
    def _draw_story_arcs(self):
        """Draw story arc information."""
        if not self.world_system.story_arcs:
            self.ax_arcs.text(0.5, 0.5, 'No story arcs yet', 
                            transform=self.ax_arcs.transAxes, ha='center', va='center',
                            color='white', fontsize=9)
            return
        
        # Count arc types
        arc_types = {}
        for arc in self.world_system.story_arcs.values():
            arc_types[arc.arc_type] = arc_types.get(arc.arc_type, 0) + 1
        
        types = list(arc_types.keys())
        counts = list(arc_types.values())
        colors = ['#ff6b6b', '#4ecdc4', '#feca57', '#ff9ff3', '#54a0ff'][:len(types)]
        
        self.ax_arcs.pie(counts, labels=types, colors=colors, autopct='%1.0f%%',
                       startangle=90, textprops={'fontsize': 8, 'color': 'white'})
        
        self.ax_arcs.set_title(f'Story Arcs ({sum(counts)} total)', color='white', fontsize=10)
    
    def _draw_timeline(self):
        """Draw comprehensive timeline of world evolution."""
        stats = self.world_system.get_world_stats()
        self.stats_history.append(stats)
        
        if len(self.stats_history) > 1:
            generations = [s['generation'] for s in self.stats_history]
            
            # Plot various metrics
            num_elements = [s['num_elements'] for s in self.stats_history]
            num_relationships = [s['num_relationships'] for s in self.stats_history]
            num_arcs = [s['num_story_arcs'] for s in self.stats_history]
            world_density = [s['world_density'] * 50 for s in self.stats_history]  # Scale for visibility
            
            self.ax_timeline.plot(generations, num_elements, label='Elements', 
                                color='#ff6b6b', linewidth=2, marker='o', markersize=3)
            self.ax_timeline.plot(generations, num_relationships, label='Relationships',
                                color='#4ecdc4', linewidth=2, marker='s', markersize=3)
            self.ax_timeline.plot(generations, num_arcs, label='Story Arcs',
                                color='#feca57', linewidth=2, marker='^', markersize=3)
            self.ax_timeline.plot(generations, world_density, label='World Density ×50',
                                color='#888888', linewidth=1, alpha=0.7)
        
        self.ax_timeline.set_xlabel('Generation', color='white')
        self.ax_timeline.set_ylabel('Count', color='white')
        self.ax_timeline.set_title('World Evolution Timeline', color='white', fontsize=12)
        self.ax_timeline.legend(fontsize=9, facecolor='#333333', edgecolor='white')
        self.ax_timeline.grid(True, color='#333333', alpha=0.3)
        self.ax_timeline.tick_params(colors='white')
    
    def run_world_generation_demo(self, max_generations: int = 200, steps_per_second: float = 4):
        """Run the world generation demonstration."""
        generation_count = 0
        
        def animate(frame):
            nonlocal generation_count
            if generation_count < max_generations:
                # World evolution
                self.world_system.step_ca()
                self.world_system.detect_and_generate_narrative_elements()
                self.world_system.evolve_relationships()
                
                # Update visualization
                self.update_visualization()
                
                # Progress logging
                if generation_count % 25 == 0:
                    stats = self.world_system.get_world_stats()
                    print(f"Gen {stats['generation']}: {stats['num_elements']} elements, "
                          f"{stats['num_relationships']} relationships, {stats['num_story_arcs']} arcs")
                    
                    # Print some narrative elements if they exist
                    if stats['num_elements'] > 0 and generation_count % 50 == 0:
                        print("\\nSample Narrative Elements:")
                        for i, (eid, element) in enumerate(list(self.world_system.narrative_elements.items())[:3]):
                            print(f"  {element.name} ({element.element_type}): {element.backstory[0] if element.backstory else 'No backstory yet'}")
                        
                        if self.world_system.relationships:
                            print("\\nSample Relationships:")
                            for i, rel in enumerate(list(self.world_system.relationships.values())[:2]):
                                e1 = self.world_system.narrative_elements.get(rel.source_id)
                                e2 = self.world_system.narrative_elements.get(rel.target_id)
                                if e1 and e2 and rel.story_events:
                                    print(f"  {rel.story_events[0]}")
                        print()
                
                generation_count += 1
            return []
        
        interval = int(1000 / steps_per_second)
        ani = animation.FuncAnimation(self.fig, animate, interval=interval, 
                                    blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
        return ani

def main(headless_generations: Optional[int] = None):
    if headless_generations:
        print("Running in headless mode...")
        world = WorldBuildingSystem(world_size=(100, 100))
        world.initialize_world(density=0.3, seed=42)
        world.run_headless(headless_generations)
        world_data = world.get_world_data()
        
        import json
        # Custom JSON encoder for sets
        class SetEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, set):
                    return list(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        print("\n--- WORLD DATA (JSON) ---")
        print(json.dumps(world_data, indent=2, cls=SetEncoder))

    else:
        print("Narrative World Generation using CA-Graph Emergence")
        print("=" * 55)
        print("Features:")
        print("- CA patterns become narrative elements (characters, locations, factions, etc.)")
        print("- Spatial proximity creates relationships and story connections")
        print("- Regional biases influence what types of elements spawn")
        print("- Story arcs emerge from relationship patterns")
        print("- Dynamic backstories and properties for all elements")
        print("- Multi-layered visualization of world, relationships, and stories")
        print()
        
        # Create world system
        world = WorldBuildingSystem(world_size=(100, 100))
        world.initialize_world(density=0.3, seed=42)
        
        # Create visualizer
        viz = NarrativeWorldVisualizer(world)
        
        print("World regions initialized:")
        for region_name, region_data in world.regions.items():
            biases = ', '.join(region_data['element_bias'])
            print(f"  {region_name.title()}: tends to spawn {biases}")
        print()
        
        try:
            print("Starting narrative world generation...")
            print("Watch for:")
            print("- Narrative elements emerging from CA patterns")
            print("- Relationships forming between nearby elements")
            print("- Story arcs developing from relationship clusters")
            print("- Regional characteristics influencing element types")
            viz.run_world_generation_demo(max_generations=150, steps_per_second=3)
        except KeyboardInterrupt:
            print("\nWorld generation stopped by user")
        except Exception as e:
            print(f"Demo error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--headless':
        generations = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        main(headless_generations=generations)
    else:
        main()