"""
Pattern Detection for CA Audio Synthesis.

Detects non-trivial patterns (gliders, oscillators, etc.) in CA
to trigger musical events (melodic hits, rhythmic patterns, etc.)
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum
from scipy import ndimage
from collections import deque


class PatternType(str, Enum):
    """Types of detectable patterns."""
    GLIDER = "glider"           # Moving pattern
    OSCILLATOR = "oscillator"   # Periodic pattern
    STILL_LIFE = "still_life"  # Stable pattern
    SPACESHIP = "spaceship"    # Moving pattern (different speeds)
    PUFFER = "puffer"          # Moving pattern leaving debris
    RAKET = "raket"             # Puffer with oscillator
    GROWTH = "growth"           # Expanding pattern
    DECAY = "decay"              # Shrinking pattern
    CHAOS = "chaos"              # Random-looking
    UNKNOWN = "unknown"


@dataclass
class DetectedPattern:
    """A detected pattern in the CA."""
    pattern_type: PatternType
    center: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # y_min, x_min, y_max, x_max
    cells: Set[Tuple[int, int]]
    confidence: float  # 0-1
    age: int  # generations since detection
    velocity: Optional[Tuple[int, int]] = None  # (dy, dx) per generation


class PatternDetector:
    """
    Detects non-trivial patterns in CA for audio triggering.
    
    Detects:
    - Still life patterns (blocks, beehives, loaves, boats, ships, ponds)
    - Oscillators (blinkers, toads, beacons, pulsars, etc.)
    - Gliders and spaceships
    - Growing/shrinking patterns
    """
    
    # Known still life patterns (as relative coordinates)
    STILL_LIFE_PATTERNS = {
        "block": {(0, 0), (0, 1), (1, 0), (1, 1)},
        "beehive": {(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2)},
        "loaf": {(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2), (3, 2)},
        "boat": {(0, 0), (0, 2), (1, 0), (1, 1), (2, 1)},
        "ship": {(0, 0), (0, 2), (1, 0), (1, 1), (2, 1)},
        "pond": {(0, 0), (0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2), (3, 3)},
    }
    
    # Known oscillator patterns (as relative coordinates, normalized)
    OSCILLATOR_PATTERNS = {
        "blinker": {(0, 0), (0, 1), (0, 2)},
        "toad": {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3)},
        "beacon": {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3)},
        "pulsar": None,  # Too complex, detect by period
    }
    
    def __init__(self):
        self.pattern_history = deque(maxlen=50)
        self.detected_patterns: List[DetectedPattern] = []
        self.previous_grid = None
        self.generation = 0
    
    def detect(
        self,
        grid: np.ndarray,
        min_pattern_size: int = 3,
        max_pattern_size: int = 30
    ) -> List[DetectedPattern]:
        """
        Detect all non-trivial patterns in the grid.
        
        Returns list of detected patterns with type and confidence.
        """
        # Find connected components
        labeled, num_features = ndimage.label(grid)
        
        patterns = []
        
        for component_id in range(1, num_features + 1):
            component_mask = (labeled == component_id)
            coords = np.argwhere(component_mask)
            
            if len(coords) < min_pattern_size or len(coords) > max_pattern_size:
                continue
            
            # Get bounding box
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Extract local pattern
            h = y_max - y_min + 1
            w = x_max - x_min + 1
            local = np.zeros((h, w), dtype=np.uint8)
            for y, x in coords:
                local[y - y_min, x - x_min] = 1
            
            # Detect pattern type
            pattern_type, confidence = self._classify_pattern(
                local, coords, grid.shape
            )
            
            # Calculate center
            center = (int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1])))
            
            # Get velocity if we have history
            velocity = None
            if self.previous_grid is not None:
                velocity = self._estimate_velocity(
                    coords, y_min, x_min
                )
            
            pattern = DetectedPattern(
                pattern_type=pattern_type,
                center=center,
                bounding_box=(y_min, x_min, y_max, x_max),
                cells=set(tuple(c) for c in coords),
                confidence=confidence,
                age=0,
                velocity=velocity
            )
            
            patterns.append(pattern)
        
        # Update history
        self.pattern_history.append(grid)
        self.previous_grid = grid.copy()
        self.generation += 1
        
        # Update pattern ages
        for p in self.detected_patterns:
            p.age += 1
        
        self.detected_patterns = patterns
        return patterns
    
    def _classify_pattern(
        self,
        local: np.ndarray,
        coords: np.ndarray,
        grid_shape: Tuple[int, int]
    ) -> Tuple[PatternType, float]:
        """Classify pattern type."""
        h, w = local.shape
        
        # Check for still life
        for name, pattern in self.STILL_LIFE_PATTERNS.items():
            if self._match_pattern(local, pattern):
                return PatternType.STILL_LIFE, 0.95
        
        # Check for oscillators
        for name, pattern in self.OSCILLATOR_PATTERNS.items():
            if pattern and self._match_pattern(local, pattern):
                return PatternType.OSCILLATOR, 0.85
        
        # Check for gliders/movement
        if self.previous_grid is not None and len(coords) > 0:
            velocity = self._estimate_velocity(
                coords, coords[:, 0].min(), coords[:, 1].min()
            )
            if velocity and (abs(velocity[0]) > 0 or abs(velocity[1]) > 0):
                # Check if pattern persists
                if len(coords) <= 10:
                    return PatternType.GLIDER, 0.7
                else:
                    return PatternType.SPACESHIP, 0.7
        
        # Check for growth/decay
        if self.previous_grid is not None:
            current_count = len(coords)
            prev_count = self._count_near_position(
                self.previous_grid, coords[0], 10
            )
            
            if current_count > prev_count * 1.5:
                return PatternType.GROWTH, 0.6
            elif current_count < prev_count * 0.5:
                return PatternType.DECAY, 0.6
        
        # Default to unknown
        return PatternType.UNKNOWN, 0.3
    
    def _match_pattern(self, local: np.ndarray, pattern: Set) -> bool:
        """Check if local pattern matches known pattern."""
        h, w = local.shape
        
        # Try different positions
        for y_offset in range(max(1, h - 5)):
            for x_offset in range(max(1, w - 5)):
                match = True
                for dy, dx in pattern:
                    py, px = y_offset + dy, x_offset + dx
                    if py < 0 or py >= h or px < 0 or px >= w:
                        match = False
                        break
                    if local[py, px] != 1:
                        match = False
                        break
                
                if match:
                    return True
        
        return False
    
    def _estimate_velocity(
        self,
        coords: np.ndarray,
        y_offset: int,
        x_offset: int
    ) -> Optional[Tuple[int, int]]:
        """Estimate pattern velocity from history."""
        if len(self.pattern_history) == 0:
            return None
        
        prev_grid = self.pattern_history[-1]
        
        # Find matching cells in previous frame
        best_velocity = (0, 0)
        best_matches = 0
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                shifted_coords = [(c[0] + dy, c[1] + dx) for c in coords]
                
                matches = sum(
                    1 for y, x in shifted_coords
                    if 0 <= y < prev_grid.shape[0] and 0 <= x < prev_grid.shape[1]
                    and prev_grid[y, x] == 1
                )
                
                if matches > best_matches:
                    best_matches = matches
                    best_velocity = (dy, dx)
        
        return best_velocity if best_matches > len(coords) * 0.5 else None
    
    def _count_near_position(
        self,
        grid: np.ndarray,
        pos: Tuple[int, int],
        radius: int
    ) -> int:
        """Count cells near a position."""
        y, x = pos
        y_min = max(0, y - radius)
        y_max = min(grid.shape[0], y + radius)
        x_min = max(0, x - radius)
        x_max = min(grid.shape[1], x + radius)
        
        return np.sum(grid[y_min:y_max, x_min:x_max])
    
    def get_audio_triggers(
        self,
        patterns: List[DetectedPattern],
        sample_rate: int = 44100
    ) -> Dict[str, List[float]]:
        """
        Convert detected patterns to audio triggers.
        
        Returns dict with trigger times for different event types.
        """
        triggers = {
            "note_on": [],      # (time, pitch, velocity)
            "note_off": [],     # (time, pitch)
            "rhythm": [],       # (time, drum_id)
            "cc": [],           # (time, controller, value)
        }
        
        BPM = 120
        beat_samples = int(sample_rate * 60 / BPM)
        
        for pattern in patterns:
            if pattern.confidence < 0.3:
                continue
            
            # Map pattern to musical parameters
            if pattern.pattern_type == PatternType.GLIDER:
                # Glider = melody note
                # Use center position to determine pitch
                pitch = 60 + (pattern.center[0] % 12)  # C major scale
                triggers["note_on"].append((
                    self.generation * beat_samples / 4,
                    pitch,
                    int(pattern.confidence * 127)
                ))
                triggers["note_off"].append((
                    self.generation * beat_samples / 4 + beat_samples,
                    pitch
                ))
            
            elif pattern.pattern_type == PatternType.OSCILLATOR:
                # Oscillator = rhythmic pattern
                triggers["rhythm"].append((
                    self.generation * beat_samples / 4,
                    0  # Kick drum
                ))
            
            elif pattern.pattern_type == PatternType.GROWTH:
                # Growth = ascending pattern
                pitch = 48 + (pattern.age % 24)
                triggers["note_on"].append((
                    self.generation * beat_samples / 8,
                    pitch,
                    100
                ))
            
            elif pattern.pattern_type == PatternType.DECAY:
                # Decay = descending pattern
                pitch = 72 - (pattern.age % 24)
                triggers["note_on"].append((
                    self.generation * beat_samples / 8,
                    pitch,
                    80
                ))
            
            elif pattern.pattern_type == PatternType.STILL_LIFE:
                # Still life = drone/continuous tone
                pitch = 36 + (pattern.center[1] % 12)  # Bass notes
                triggers["cc"].append((
                    self.generation * beat_samples / 4,
                    1,  # Mod wheel
                    int(pattern.confidence * 127)
                ))
        
        return triggers


class DynamicPatternDetector(PatternDetector):
    """Extended detector for dynamic/evolving patterns."""
    
    def __init__(self):
        super().__init__()
        self.activity_history = deque(maxlen=100)
        self.density_history = deque(maxlen=100)
    
    def analyze_dynamics(self, grid: np.ndarray) -> Dict[str, float]:
        """Analyze dynamic properties of the CA."""
        # Current metrics
        density = np.mean(grid)
        activity = self._calculate_activity(grid)
        
        # Store in history
        self.density_history.append(density)
        self.activity_history.append(activity)
        
        # Calculate trends
        density_trend = 0.0
        activity_trend = 0.0
        
        if len(self.density_history) >= 10:
            recent = list(self.density_history)[-10:]
            density_trend = (recent[-1] - recent[0]) / len(recent)
            
            recent = list(self.activity_history)[-10:]
            activity_trend = (recent[-1] - recent[0]) / len(recent)
        
        return {
            "density": density,
            "activity": activity,
            "density_trend": density_trend,
            "activity_trend": activity_trend,
            "is_growing": density_trend > 0.01,
            "is_dying": density_trend < -0.01,
            "is_chaotic": abs(activity_trend) > 0.1,
            "is_stable": abs(density_trend) < 0.001,
        }
    
    def _calculate_activity(self, grid: np.ndarray) -> float:
        """Calculate activity level (cells changed from previous)."""
        if self.previous_grid is None:
            return 0.5
        
        changed = np.sum(grid != self.previous_grid)
        total = grid.size
        
        return changed / total if total > 0 else 0


if __name__ == '__main__':
    print("Testing Pattern Detector...")
    
    from SynthLab.ca_engine import CAEngine, CARule
    
    # Create CA and evolve
    ca = CAEngine((50, 50), CARule.CONWAY, seed=42)
    ca.initialize_random(0.3)
    
    detector = DynamicPatternDetector()
    
    # Detect patterns over time
    print("\nDetecting patterns...")
    for gen in range(50):
        patterns = detector.detect(ca.grid)
        dynamics = detector.analyze_dynamics(ca.grid)
        
        if patterns:
            pattern_types = set(p.pattern_type.value for p in patterns)
            print(f"Gen {gen}: {len(patterns)} patterns - {pattern_types}")
            print(f"  Dynamics: density={dynamics['density']:.3f}, trend={dynamics['density_trend']:.4f}")
        
        ca.step()
    
    # Get audio triggers
    triggers = detector.get_audio_triggers(detector.detected_patterns)
    print(f"\nAudio triggers: {len(triggers['note_on'])} notes, {len(triggers['rhythm'])} rhythms")
    
    print("\nPattern detection working!")
