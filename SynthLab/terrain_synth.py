"""
Terrain Synthesis Engine for AlchemicalSynth.

Combines CA and fractal terrain generation with trajectory scanning
to produce audio output.

Based on the Wave Terrain Synthesis approach from Terrain synth:
- 2D trajectory scans over 3D terrain surface
- Trajectory parameters can be modified at audio rate
- Terrain can be generated via CA, fractals, or hybrid
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass

from SynthLab.ca_engine import CAEngine, CARule, HexCAEngine
from SynthLab.fractal_generator import FractalGenerator, FractalType


class TrajectoryShape(str, Enum):
    """2D trajectory shapes."""
    ELLIPSE = "ellipse"
    FIGURE8 = "figure8"
    LISSABOUS = "lissajous"
    MEANDER = "meander"
    SPIRAL = "spiral"
    ROSE = "rose"
    CUSTOM = "custom_path"


@dataclass
class TrajectoryState:
    """Current state of the trajectory."""
    x: float  # Normalized position [0, 1]
    y: float
    phase: float  # Phase in radians
    velocity_x: float
    velocity_y: float


class TerrainSynthesizer:
    """
    Main terrain synthesis engine.
    
    Combines:
    - CA or fractal terrain generation
    - 2D trajectory scanning
    - Waveform generation
    - Envelope and filter processing
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 256
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # Terrain systems
        self.ca_engine: Optional[CAEngine] = None
        self.hex_ca: Optional[HexCAEngine] = None
        self.fractal_gen: Optional[FractalGenerator] = None
        
        # Terrain parameters
        self.terrain_type = "ca_fractal_hybrid"
        self.grid_size = (128, 128)
        self.hybrid_blend = 0.5
        
        # Trajectory state
        self.trajectory_state = TrajectoryState(
            x=0.5, y=0.5, phase=0.0,
            velocity_x=0.0, velocity_y=0.0
        )
        
        # Audio processing state
        self.envelope_state = 0.0
        self.envelope_stage = "idle"  # idle, attack, decay, sustain, release
        self.envelope_counter = 0
        
        # Feedback buffer
        self.feedback_buffer = np.zeros(44100)  # 1 second buffer
        self.feedback_position = 0
        self.feedback_enabled = False
        
        # LFO state
        self.lfo_phase = 0.0
        
        # Initialize default terrain
        self._init_default_terrain()
    
    def _init_default_terrain(self):
        """Initialize default terrain generators."""
        size = self.grid_size
        
        # CA engine
        self.ca_engine = CAEngine(
            size=size,
            rule=CARule.CONWAY,
            seed=42
        )
        self.ca_engine.initialize_random(density=0.3)
        
        # Fractal generator
        self.fractal_gen = FractalGenerator(size=size, seed=42)
    
    def set_terrain_type(self, terrain_type: str):
        """Set terrain generation type."""
        self.terrain_type = terrain_type
    
    def set_ca_rule(self, rule: str):
        """Set CA rule."""
        if self.ca_engine is None:
            self.ca_engine = CAEngine(self.grid_size, CARule(rule))
        else:
            self.ca_engine.rule = CARule(rule)
    
    def set_fractal_type(self, fractal_type: str):
        """Set fractal type."""
        # Already handled in generate_terrain
        pass
    
    def generate_terrain(self, fractal_type: str = "perlin") -> np.ndarray:
        """
        Generate terrain heightmap.
        
        Returns 2D array of terrain heights in [0, 1].
        """
        terrain = np.zeros(self.grid_size, dtype=np.float64)
        
        if self.terrain_type in ["ca", "ca_fractal_hybrid"]:
            # Generate CA terrain
            ca_terrain = self.ca_engine.get_terrain_height()
            
            if self.terrain_type == "ca_fractal_hybrid":
                # Get fractal terrain
                frac_terrain = self.fractal_gen.hybrid(
                    fractal_types=[fractal_type, "worley", "ridged"],
                    blend_weights=[0.5, 0.3, 0.2]
                )
                
                # Blend
                terrain = (1 - self.hybrid_blend) * ca_terrain + self.hybrid_blend * frac_terrain
            else:
                terrain = ca_terrain
        
        elif self.terrain_type == "fractal":
            terrain = self.fractal_gen.hybrid(
                fractal_types=[fractal_type, "worley", "ridged"],
                blend_weights=[0.5, 0.3, 0.2]
            )
        
        elif self.terrain_type == "hex":
            if self.hex_ca is None:
                self.hex_ca = HexCAEngine(radius=30)
                self.hex_ca.initialize_random(0.3)
            
            hex_arr = self.hex_ca.to_array()
            # Resize to match grid_size
            terrain = np.zeros(self.grid_size, dtype=np.float64)
            h_min = min(terrain.shape[0], hex_arr.shape[0])
            w_min = min(terrain.shape[1], hex_arr.shape[1])
            terrain[:h_min, :w_min] = hex_arr[:h_min, :w_min]
        
        return terrain
    
    def step_ca(self, steps: int = 1):
        """Evolve CA by n steps."""
        if self.ca_engine:
            self.ca_engine.step_n(steps)
        if self.hex_ca:
            for _ in range(steps):
                self.hex_ca.step()
    
    # ==================== TRAJECTORY ====================
    
    def compute_trajectory_position(
        self,
        shape: str,
        phase: float,
        params: dict
    ) -> Tuple[float, float]:
        """
        Get 2D position on trajectory at given phase.
        
        Returns (x, y) in normalized [0, 1] range.
        """
        shape = TrajectoryShape(shape)
        
        if shape == TrajectoryShape.ELLIPSE:
            # Basic elliptical orbit
            rx = params.get("radius_x", 0.4)
            ry = params.get("radius_y", 0.4)
            cx = params.get("center_x", 0.5)
            cy = params.get("center_y", 0.5)
            return (
                cx + rx * np.cos(phase),
                cy + ry * np.sin(phase)
            )
        
        elif shape == TrajectoryShape.FIGURE8:
            # Lemniscate (figure-8)
            scale = params.get("scale", 0.3)
            center = params.get("center", (0.5, 0.5))
            t = phase / (2 * np.pi)
            x = center[0] + scale * np.sin(2 * t)
            y = center[1] + scale * np.sin(t)
            return (x, y)
        
        elif shape == TrajectoryShape.LISSABOUS:
            # Lissajous curve
            ax = params.get("a", 3)
            ay = params.get("b", 2)
            delta = params.get("delta", np.pi / 2)
            cx, cy = params.get("center", (0.5, 0.5))
            scale = params.get("scale", 0.3)
            return (
                cx + scale * np.sin(ax * phase + delta),
                cy + scale * np.sin(ay * phase)
            )
        
        elif shape == TrajectoryShape.SPIRAL:
            # Archimedean spiral
            a = params.get("a", 0.1)
            b = params.get("b", 0.05)
            cx, cy = params.get("center", (0.5, 0.5))
            r = a + b * phase
            return (
                cx + r * np.cos(phase),
                cy + r * np.sin(phase)
            )
        
        elif shape == TrajectoryShape.ROSE:
            # Rose curve
            k = params.get("k", 3)  # petals
            r_max = params.get("scale", 0.4)
            cx, cy = params.get("center", (0.5, 0.5))
            r = r_max * np.cos(k * phase)
            return (
                cx + r * np.cos(phase),
                cy + r * np.sin(phase)
            )
        
        elif shape == TrajectoryShape.MEANDER:
            # Random walk with mean reversion
            speed = params.get("speed", 0.01)
            meanderance = params.get("meanderance", 0.2)
            
            # Update position with noise
            self.trajectory_state.x += np.random.normal(0, meanderance)
            self.trajectory_state.y += np.random.normal(0, meanderance)
            
            # Mean reversion
            self.trajectory_state.x += (0.5 - self.trajectory_state.x) * speed
            self.trajectory_state.y += (0.5 - self.trajectory_state.y) * speed
            
            # Clamp to bounds
            x = np.clip(self.trajectory_state.x, 0, 1)
            y = np.clip(self.trajectory_state.y, 0, 1)
            return (x, y)
        
        else:
            # Default ellipse
            return (0.5 + 0.3 * np.cos(phase), 0.5 + 0.3 * np.sin(phase))
    
    # ==================== AUDIO SYNTHESIS ====================
    
    def sample_terrain(self, terrain: np.ndarray, x: float, y: float) -> float:
        """
        Sample terrain at continuous (x, y) position.
        
        Uses bilinear interpolation.
        """
        h, w = terrain.shape
        
        # Convert to pixel coordinates
        px = x * (w - 1)
        py = y * (h - 1)
        
        # Get integer and fractional parts
        x0 = int(np.floor(px))
        y0 = int(np.floor(py))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        
        fx = px - x0
        fy = py - y0
        
        # Bilinear interpolation
        v00 = terrain[y0, x0]
        v10 = terrain[y0, x1]
        v01 = terrain[y1, x0]
        v11 = terrain[y1, x1]
        
        v0 = v00 * (1 - fx) + v10 * fx
        v1 = v01 * (1 - fx) + v11 * fx
        
        return v0 * (1 - fy) + v1 * fy
    
    def generate_sample(
        self,
        terrain: np.ndarray,
        trajectory_params: dict,
        waveform: str = "sine"
    ) -> float:
        """
        Generate a single audio sample.
        
        Uses terrain height to modulate waveform.
        """
        # Get trajectory position
        phase = self.trajectory_state.phase
        x, y = self.compute_trajectory_position(
            trajectory_params.get("shape", "ellipse"),
            phase,
            trajectory_params
        )
        
        # Sample terrain
        z = self.sample_terrain(terrain, x, y)
        
        # Base frequency from terrain
        base_freq = trajectory_params.get("frequency", 220)
        
        # Get harmonics
        harmonics = trajectory_params.get("harmonics", [1, 2, 3, 5])
        harmonic_amps = trajectory_params.get("harmonic_amps", [1.0, 0.5, 0.25, 0.125])
        
        # Generate waveform
        sample = 0.0
        t = phase / (2 * np.pi)  # Normalized phase
        
        for h, amp in zip(harmonics, harmonic_amps):
            freq = base_freq * h
            
            if waveform == "sine":
                sample += amp * np.sin(2 * np.pi * freq * t)
            elif waveform == "saw":
                sample += amp * (2 * (t * h % 1) - 1)
            elif waveform == "square":
                sample += amp * (1 if (t * h % 1) > 0.5 else -1)
            elif waveform == "triangle":
                sample += amp * (4 * abs(t * h % 1 - 0.5) - 1)
            elif waveform == "sample_hold":
                sample += amp * (1 if (t * h % 1) > z else -1)
        
        # Apply terrain as amplitude modulation
        sample *= 0.5 + z * 0.5
        
        # Apply envelope
        sample *= self.envelope_state
        
        # Feedback processing
        if self.feedback_enabled and trajectory_params.get("feedback", {}).get("enabled", False):
            fb = trajectory_params.get("feedback", {})
            delay_samples = fb.get("delay_samples", 4410)
            compression = fb.get("compression", 0.8)
            
            # Read from feedback buffer
            fb_idx = (self.feedback_position - delay_samples) % len(self.feedback_buffer)
            fb_sample = self.feedback_buffer[fb_idx]
            
            # Mix feedback
            sample = sample * 0.7 + fb_sample * compression * 0.3
            
            # Write to feedback buffer
            self.feedback_buffer[self.feedback_position] = sample
            self.feedback_position = (self.feedback_position + 1) % len(self.feedback_buffer)
        
        return np.clip(sample, -1, 1)
    
    def generate_block(
        self,
        terrain: np.ndarray,
        trajectory_params: dict,
        num_samples: int = 256
    ) -> np.ndarray:
        """Generate a block of audio samples."""
        samples = np.zeros(num_samples)
        
        # Phase increment per sample
        freq = trajectory_params.get("frequency", 220)
        phase_increment = freq / self.sample_rate
        
        for i in range(num_samples):
            samples[i] = self.generate_sample(terrain, trajectory_params)
            
            # Update phase
            self.trajectory_state.phase += phase_increment
            
            # Apply translation
            trans = trajectory_params.get("translation", {})
            if trans:
                self.trajectory_state.x += trans.get("x_speed", 0) / self.sample_rate
                self.trajectory_state.y += trans.get("y_speed", 0) / self.sample_rate
                
                # Circular translation
                if trans.get("circular", False):
                    angle = self.trajectory_state.x * 2 * np.pi
                    radius = 0.2
                    self.trajectory_state.x = 0.5 + radius * np.cos(angle)
                    self.trajectory_state.y = 0.5 + radius * np.sin(angle)
                
                # Clamp
                self.trajectory_state.x = np.clip(self.trajectory_state.x, 0, 1)
                self.trajectory_state.y = np.clip(self.trajectory_state.y, 0, 1)
            
            # Update envelope
            self._update_envelope(trajectory_params.get("envelope", {}))
            
            # Update LFO
            self._update_lfo(trajectory_params.get("modulation", {}))
        
        # Apply filter
        samples = self._apply_filter(
            samples,
            trajectory_params.get("filter", {})
        )
        
        # Apply saturation
        samples = self._apply_saturation(
            samples,
            trajectory_params.get("saturation", 0)
        )
        
        return samples
    
    def _update_envelope(self, env_params: dict):
        """Update envelope state."""
        attack = env_params.get("attack", 0.01)
        decay = env_params.get("decay", 0.2)
        sustain = env_params.get("sustain", 0.7)
        release = env_params.get("release", 0.5)
        
        if self.envelope_stage == "idle":
            pass
        elif self.envelope_stage == "attack":
            rate = 1.0 / (attack * self.sample_rate)
            self.envelope_state += rate
            if self.envelope_state >= 1.0:
                self.envelope_state = 1.0
                self.envelope_stage = "decay"
        elif self.envelope_stage == "decay":
            rate = 1.0 / (decay * self.sample_rate)
            self.envelope_state -= rate * (1 - sustain)
            if self.envelope_state <= sustain:
                self.envelope_state = sustain
                self.envelope_stage = "sustain"
        elif self.envelope_stage == "sustain":
            pass
        elif self.envelope_stage == "release":
            rate = 1.0 / (release * self.sample_rate)
            self.envelope_state -= rate
            if self.envelope_state <= 0:
                self.envelope_state = 0
                self.envelope_stage = "idle"
    
    def _update_lfo(self, mod_params: dict):
        """Update LFO for modulation."""
        lfo_freq = mod_params.get("lfo_frequency", 1)
        lfo_depth = mod_params.get("lfo_depth", 0.3)
        
        if lfo_freq > 0:
            self.lfo_phase += 2 * np.pi * lfo_freq / self.sample_rate
            self.lfo_phase = self.lfo_phase % (2 * np.pi)
    
    def _apply_filter(self, samples: np.ndarray, filter_params: dict) -> np.ndarray:
        """Apply lowpass filter."""
        filter_type = filter_params.get("type", "lowpass")
        
        if filter_type == "lowpass":
            # Simple one-pole lowpass
            cutoff = filter_params.get("cutoff", 8000)
            resonance = filter_params.get("resonance", 0.5)
            
            # Calculate coefficient
            rc = 1.0 / (2 * np.pi * cutoff)
            dt = 1.0 / self.sample_rate
            alpha = dt / (rc + dt)
            
            # Apply filter
            filtered = np.zeros_like(samples)
            state = samples[0]
            for i in range(len(samples)):
                state += alpha * (samples[i] - state)
                filtered[i] = state
            
            return filtered
        
        return samples
    
    def _apply_saturation(self, samples: np.ndarray, amount: float) -> np.ndarray:
        """Apply waveshaping saturation."""
        if amount > 0:
            # Soft clipping
            return np.tanh(samples * (1 + amount * 10)) / np.tanh(1 + amount * 10)
        return samples
    
    # ==================== CONTROL ====================
    
    def trigger_attack(self):
        """Start envelope attack."""
        self.envelope_stage = "attack"
    
    def trigger_release(self):
        """Start envelope release."""
        self.envelope_stage = "release"
    
    def set_preset(self, preset):
        """Apply audio preset configuration."""
        self.terrain_type = preset.terrain_type
        self.grid_size = tuple(preset.grid_size)
        self.hybrid_blend = preset.hybrid_blend
        
        # Reinitialize terrain
        self._init_default_terrain()
        
        # Set CA rule
        if hasattr(preset, 'ca_rule'):
            self.set_ca_rule(preset.ca_rule)
        
        # Set seed
        if preset.seed:
            np.random.seed(preset.seed)


class AudioEngine:
    """Complete audio engine managing multiple synthesizer instances."""
    
    def __init__(self, sample_rate: int = 44100, block_size: int = 256):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.synths: List[TerrainSynthesizer] = []
    
    def add_synth(self) -> TerrainSynthesizer:
        """Add a new synthesizer instance."""
        synth = TerrainSynthesizer(self.sample_rate, self.block_size)
        self.synths.append(synth)
        return synth
    
    def process(
        self,
        terrain: np.ndarray,
        trajectory_params: dict,
        num_samples: int = None
    ) -> np.ndarray:
        """Process audio from all synths."""
        if num_samples is None:
            num_samples = self.block_size
        
        output = np.zeros(num_samples)
        
        for synth in self.synths:
            output += synth.generate_block(terrain, trajectory_params, num_samples)
        
        # Soft clip output
        return np.tanh(output)


if __name__ == '__main__':
    print("Testing Terrain Synthesizer...")
    
    # Create synthesizer
    synth = TerrainSynthesizer()
    
    # Generate terrain
    print("Generating terrain...")
    terrain = synth.generate_terrain("perlin")
    print(f"  Terrain: shape={terrain.shape}, min={terrain.min():.3f}, max={terrain.max():.3f}")
    
    # Evolve CA a bit
    synth.step_ca(10)
    terrain = synth.generate_terrain("perlin")
    print(f"  After CA evolution: min={terrain.min():.3f}, max={terrain.max():.3f}")
    
    # Generate some audio
    print("\nGenerating audio block...")
    traj_params = {
        "shape": "ellipse",
        "frequency": 220,
        "harmonics": [1, 2, 3],
        "harmonic_amps": [1.0, 0.5, 0.25],
        "center": (0.5, 0.5),
        "radius_x": 0.3,
        "radius_y": 0.3,
        "filter": {"type": "lowpass", "cutoff": 8000},
        "envelope": {"attack": 0.01, "decay": 0.2, "sustain": 0.7, "release": 0.5},
    }
    
    synth.trigger_attack()
    audio = synth.generate_block(terrain, traj_params, 1024)
    print(f"  Audio: shape={audio.shape}, min={audio.min():.3f}, max={audio.max():.3f}")
    
    # Test different trajectory shapes
    print("\nTesting trajectory shapes...")
    shapes = ["ellipse", "figure8", "lissajous", "rose"]
    for shape in shapes:
        pos = synth.get_trajectory_position(shape, 0.0, {"scale": 0.3})
        print(f"  {shape}: {pos}")
    
    print("\nTerrain synthesizer working!")
