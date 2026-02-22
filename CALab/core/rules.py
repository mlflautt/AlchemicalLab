"""Rule systems for cellular automata based on Wolfram's classifications."""

import numpy as np
from typing import Callable, Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import json


class WolframClass(Enum):
    """Wolfram's four classes of CA behavior."""
    CLASS_1 = 1  # Uniform: Evolution leads to homogeneous state
    CLASS_2 = 2  # Periodic: Evolution leads to periodic patterns
    CLASS_3 = 3  # Chaotic: Random, aperiodic patterns
    CLASS_4 = 4  # Complex: Localized structures with complex interactions


class Rule(ABC):
    """Abstract base class for CA rules."""
    
    @abstractmethod
    def apply(self, neighbors: np.ndarray, current: int) -> int:
        """Apply rule to determine next state.
        
        Args:
            neighbors: Array of neighbor states
            current: Current cell state
            
        Returns:
            Next state
        """
        pass
    
    @abstractmethod
    def get_classification(self) -> Optional[WolframClass]:
        """Get Wolfram classification if known."""
        pass


class ElementaryRule(Rule):
    """Elementary cellular automaton rule (1D binary)."""
    
    # Known classifications for elementary rules
    CLASSIFICATIONS = {
        0: WolframClass.CLASS_1,
        1: WolframClass.CLASS_2,
        4: WolframClass.CLASS_2,
        18: WolframClass.CLASS_3,
        22: WolframClass.CLASS_3,
        30: WolframClass.CLASS_3,
        45: WolframClass.CLASS_3,
        54: WolframClass.CLASS_4,
        60: WolframClass.CLASS_4,
        62: WolframClass.CLASS_4,
        73: WolframClass.CLASS_2,
        90: WolframClass.CLASS_3,
        105: WolframClass.CLASS_3,
        110: WolframClass.CLASS_4,  # Universal computation
        122: WolframClass.CLASS_3,
        126: WolframClass.CLASS_3,
        150: WolframClass.CLASS_3,
        184: WolframClass.CLASS_2,  # Traffic flow
        254: WolframClass.CLASS_1,
    }
    
    def __init__(self, rule_number: int):
        """Initialize elementary rule.
        
        Args:
            rule_number: Wolfram rule number (0-255)
        """
        if not 0 <= rule_number <= 255:
            raise ValueError(f"Rule number must be 0-255, got {rule_number}")
        
        self.rule_number = rule_number
        self.lookup_table = self._generate_lookup_table()
        
    def _generate_lookup_table(self) -> Dict[Tuple[int, int, int], int]:
        """Generate lookup table from rule number."""
        table = {}
        binary = format(self.rule_number, '08b')
        
        for i, config in enumerate([
            (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
            (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0)
        ]):
            table[config] = int(binary[7 - i])
        
        return table
    
    def apply(self, neighbors: np.ndarray, current: int) -> int:
        """Apply elementary rule."""
        if len(neighbors) != 2:
            raise ValueError("Elementary rule requires exactly 2 neighbors")
        
        left, right = neighbors
        config = (left, current, right)
        return self.lookup_table[config]
    
    def get_classification(self) -> Optional[WolframClass]:
        """Get Wolfram classification if known."""
        return self.CLASSIFICATIONS.get(self.rule_number)
    
    def is_universal(self) -> bool:
        """Check if this rule is computationally universal."""
        return self.rule_number == 110
    
    def __repr__(self) -> str:
        """String representation."""
        classification = self.get_classification()
        class_str = f" (Class {classification.value})" if classification else ""
        universal_str = " [Universal]" if self.is_universal() else ""
        return f"ElementaryRule({self.rule_number}{class_str}{universal_str})"


class TotalisticRule(Rule):
    """Totalistic rule based on sum of neighbors."""
    
    def __init__(self, birth_values: List[int], survival_values: List[int], 
                 states: int = 2):
        """Initialize totalistic rule.
        
        Args:
            birth_values: Sums that cause birth (dead -> alive)
            survival_values: Sums that cause survival (alive -> alive)
            states: Number of states
        """
        self.birth_values = set(birth_values)
        self.survival_values = set(survival_values)
        self.states = states
        
    def apply(self, neighbors: np.ndarray, current: int) -> int:
        """Apply totalistic rule."""
        total = np.sum(neighbors)
        
        if current == 0:  # Dead cell
            return 1 if total in self.birth_values else 0
        else:  # Alive cell
            return 1 if total in self.survival_values else 0
    
    def get_classification(self) -> Optional[WolframClass]:
        """Totalistic rules don't have predetermined classifications."""
        return None
    
    def to_rulestring(self) -> str:
        """Convert to B/S notation."""
        birth_str = ''.join(str(b) for b in sorted(self.birth_values))
        survival_str = ''.join(str(s) for s in sorted(self.survival_values))
        return f"B{birth_str}/S{survival_str}"
    
    @classmethod
    def from_rulestring(cls, rulestring: str) -> 'TotalisticRule':
        """Create from B/S notation (e.g., 'B3/S23' for Game of Life)."""
        parts = rulestring.split('/')
        birth_part = parts[0][1:]  # Remove 'B'
        survival_part = parts[1][1:]  # Remove 'S'
        
        birth_values = [int(d) for d in birth_part] if birth_part else []
        survival_values = [int(d) for d in survival_part] if survival_part else []
        
        return cls(birth_values, survival_values)


class ContinuousRule(Rule):
    """Rule for continuous-valued cellular automata."""
    
    def __init__(self, update_function: Callable[[np.ndarray, float], float],
                 classification: Optional[WolframClass] = None):
        """Initialize continuous rule.
        
        Args:
            update_function: Function (neighbors, current) -> next_state
            classification: Optional Wolfram classification
        """
        self.update_function = update_function
        self.classification = classification
    
    def apply(self, neighbors: np.ndarray, current: float) -> float:
        """Apply continuous rule."""
        return self.update_function(neighbors, current)
    
    def get_classification(self) -> Optional[WolframClass]:
        """Get classification if provided."""
        return self.classification


class ProbabilisticRule(Rule):
    """Stochastic rule with probabilistic transitions."""
    
    def __init__(self, base_rule: Rule, noise_level: float = 0.01):
        """Initialize probabilistic rule.
        
        Args:
            base_rule: Underlying deterministic rule
            noise_level: Probability of random flip
        """
        self.base_rule = base_rule
        self.noise_level = noise_level
    
    def apply(self, neighbors: np.ndarray, current: int) -> int:
        """Apply rule with random perturbation."""
        result = self.base_rule.apply(neighbors, current)
        
        if np.random.random() < self.noise_level:
            # Flip the result
            result = 1 - result if result in [0, 1] else np.random.randint(0, 2)
        
        return result
    
    def get_classification(self) -> Optional[WolframClass]:
        """Classification becomes uncertain with noise."""
        base_class = self.base_rule.get_classification()
        if base_class and self.noise_level > 0.05:
            # High noise tends toward Class 3 (chaotic)
            return WolframClass.CLASS_3
        return base_class


class RuleSet:
    """Collection of rules for multi-rule systems."""
    
    def __init__(self, rules: Dict[str, Rule]):
        """Initialize rule set.
        
        Args:
            rules: Dictionary mapping rule names to Rule objects
        """
        self.rules = rules
        self.active_rule = None
        
    def select_rule(self, name: str) -> Rule:
        """Select active rule by name."""
        if name not in self.rules:
            raise ValueError(f"Rule '{name}' not found")
        self.active_rule = self.rules[name]
        return self.active_rule
    
    def add_rule(self, name: str, rule: Rule) -> None:
        """Add a new rule to the set."""
        self.rules[name] = rule
    
    def apply(self, neighbors: np.ndarray, current: int) -> int:
        """Apply the active rule."""
        if self.active_rule is None:
            raise ValueError("No active rule selected")
        return self.active_rule.apply(neighbors, current)
    
    def get_classifications(self) -> Dict[str, Optional[WolframClass]]:
        """Get classifications of all rules."""
        return {name: rule.get_classification() 
                for name, rule in self.rules.items()}


class HybridRule(Rule):
    """Combines multiple rules based on spatial or temporal conditions."""
    
    def __init__(self, rules: List[Rule], 
                 selector: Callable[[int, int], int]):
        """Initialize hybrid rule.
        
        Args:
            rules: List of rules to combine
            selector: Function (x, t) -> rule_index
        """
        self.rules = rules
        self.selector = selector
        self.time_step = 0
        
    def apply(self, neighbors: np.ndarray, current: int, 
              position: Optional[int] = None) -> int:
        """Apply selected rule based on position/time."""
        if position is None:
            position = 0
        
        rule_index = self.selector(position, self.time_step)
        rule_index = rule_index % len(self.rules)
        
        return self.rules[rule_index].apply(neighbors, current)
    
    def increment_time(self) -> None:
        """Increment internal time counter."""
        self.time_step += 1
    
    def get_classification(self) -> Optional[WolframClass]:
        """Hybrid rules typically produce complex behavior."""
        return WolframClass.CLASS_4


class LangtonsAntRule(Rule):
    """Rule for Langton's Ant and similar turmite systems."""
    
    def __init__(self, turn_rules: str = "RL"):
        """Initialize Langton's Ant rule.
        
        Args:
            turn_rules: String of L/R turns for each color
        """
        self.turn_rules = turn_rules
        self.num_colors = len(turn_rules)
        
    def apply(self, neighbors: np.ndarray, current: int) -> int:
        """This is handled differently - see LangtonsAnt implementation."""
        # Langton's Ant requires special handling with ant position tracking
        raise NotImplementedError("Use LangtonsAnt automaton class")
    
    def get_classification(self) -> Optional[WolframClass]:
        """Langton's Ant exhibits complex behavior."""
        return WolframClass.CLASS_4


class LifeLikeRule(TotalisticRule):
    """Specialized totalistic rule for Life-like automata."""
    
    # Well-known Life-like rules with their behaviors
    KNOWN_RULES = {
        "B3/S23": ("Conway's Life", WolframClass.CLASS_4),
        "B36/S23": ("HighLife", WolframClass.CLASS_4),
        "B3678/S34678": ("Day & Night", WolframClass.CLASS_3),
        "B1357/S1357": ("Replicator", WolframClass.CLASS_3),
        "B2/S": ("Seeds", WolframClass.CLASS_3),
        "B234/S": ("Serviettes", WolframClass.CLASS_2),
        "B0/S8": ("Anti-Life", WolframClass.CLASS_2),
    }
    
    def __init__(self, rulestring: str):
        """Initialize from rulestring."""
        super().__init__(*self._parse_rulestring(rulestring))
        self.rulestring = rulestring
        self.name, self.classification = self.KNOWN_RULES.get(
            rulestring, ("Custom", None)
        )
    
    @staticmethod
    def _parse_rulestring(rulestring: str) -> Tuple[List[int], List[int]]:
        """Parse B/S notation."""
        parts = rulestring.split('/')
        birth_part = parts[0][1:]
        survival_part = parts[1][1:] if len(parts) > 1 else ""
        
        birth = [int(d) for d in birth_part] if birth_part else []
        survival = [int(d) for d in survival_part] if survival_part else []
        
        return birth, survival
    
    def get_classification(self) -> Optional[WolframClass]:
        """Get known classification."""
        return self.classification


def create_world_building_rules() -> RuleSet:
    """Create a set of rules optimized for world-building.
    
    Returns different rules for different world aspects:
    - Terrain generation (Class 2/3)
    - Civilization growth (Class 4)
    - Resource distribution (Class 3)
    - Climate patterns (Class 3/4)
    """
    rules = {
        # Stable terrain features
        "mountains": ElementaryRule(254),  # Class 1 - stable
        
        # Dynamic weather/climate
        "weather": ElementaryRule(30),  # Class 3 - chaotic
        
        # Complex civilization dynamics
        "civilization": LifeLikeRule("B3/S23"),  # Class 4 - complex
        
        # Resource distribution
        "resources": ElementaryRule(90),  # Class 3 - pseudo-random
        
        # Border dynamics
        "borders": ElementaryRule(110),  # Class 4 - universal computation
        
        # Trade routes (modified Life)
        "trade": LifeLikeRule("B36/S23"),  # HighLife - traveling patterns
        
        # Conflict zones
        "conflict": LifeLikeRule("B234/S"),  # Serviettes - explosive growth
        
        # Cultural spread
        "culture": ProbabilisticRule(
            LifeLikeRule("B3/S23"), 
            noise_level=0.02
        ),  # Life with noise
    }
    
    return RuleSet(rules)