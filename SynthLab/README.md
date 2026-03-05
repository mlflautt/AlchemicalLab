# AlchemicalSynth - Wave Terrain Synthesis System

A generative audio synthesis system combining Cellular Automata (CA), Fractals, and Neural networks for creating evolving soundscapes. Designed for both standalone Python usage and as a JUCE-based AU/VST3 plugin.

## Quick Start

### Python API

```python
from SynthLab.terrain_synth import TerrainSynthesizer

# Create synthesizer
synth = TerrainSynthesizer(sample_rate=44100)

# Generate terrain (CA + Fractal hybrid)
terrain = synth.generate_terrain('perlin')

# Evolve CA over time
synth.step_ca(10)
terrain = synth.generate_terrain('perlin')

# Configure trajectory
traj_params = {
    'shape': 'ellipse',
    'frequency': 220,
    'harmonics': [1, 2, 3, 5],
    'harmonic_amps': [1.0, 0.5, 0.25, 0.125],
    'filter': {'type': 'lowpass', 'cutoff': 8000},
    'envelope': {'attack': 0.01, 'decay': 0.2, 'sustain': 0.7, 'release': 0.5},
}

# Generate audio
synth.trigger_attack()
audio = synth.generate_block(terrain, traj_params, 1024)
# audio now contains 1024 samples of generated audio
```

### Using Presets

```python
from SynthLab.audio_preset import AudioPreset, create_default_preset

# Load preset from file
preset = AudioPreset.from_file('path/to/preset.json')

# Or create default
preset = create_default_preset()

# Apply to synthesizer
synth.set_preset(preset)
```

### Pattern Detection for Generative Music

```python
from SynthLab.ca_engine import CAEngine, CARule
from SynthLab.pattern_detector import DynamicPatternDetector

# Create CA
ca = CAEngine((50, 50), CARule.CONWAY, seed=42)
ca.initialize_random(0.3)

# Create pattern detector
detector = DynamicPatternDetector()

# Detect patterns over generations
for gen in range(100):
    patterns = detector.detect(ca.grid)
    dynamics = detector.analyze_dynamics(ca.grid)
    
    # Get audio triggers
    triggers = detector.get_audio_triggers(patterns)
    # triggers contains 'note_on', 'note_off', 'rhythm', 'cc' events
    
    ca.step()
```

## Current Features

### CA Rules (16 types)
- **Conway's Game of Life** - Classic chaotic patterns
- **Brian's Brain** - 3-state wave patterns  
- **HighLife** - Like Conway with 6-neighbor birth
- **Day & Night** - Symmetric rich structures
- **Seeds** - Explosive growth
- **Replicator** - Pattern replication
- **Morley** - Moving ships/puffers
- **Rule 30/90/110** - 1D CA (chaotic/fractal/Turing complete)
- **And more...**

### Fractal Types (10+ types)
- **Perlin Noise** - Smooth organic textures
- **Simplex Noise** - Faster, less artifacts
- **Worley/Cellular** - Cell-like patterns
- **Mandelbrot** - Mathematical landscapes
- **Julia Sets** - Parametric variations
- **FBM** - Fractal Brownian Motion
- **Ridged Multifractal** - Sharp ridges
- **Domain Warping** - Flowing patterns

### Trajectory Shapes (6 types)
- **Ellipse** - Basic orbital scan
- **Figure-8 (Lemniscate)** - Figure-8 path
- **Lissajous** - Complex periodic curves
- **Spiral** - Archimedean spiral
- **Rose** - Rose curve petals
- **Meander** - Random walk with mean reversion

### Audio Processing
- Multi-oscillator synthesis (sine, saw, square, triangle, sample-hold)
- Harmonic series with custom amplitudes
- ADSR envelope with ES (envelope-size) toggle
- Lowpass filter with resonance
- Soft clipping saturation
- Feedback processing
- LFO modulation

### Pattern Detection
- Glider detection → melodic triggers
- Oscillator detection → rhythmic patterns
- Growth/decay → pitch bends
- Still life → drone/ambient

### Persistence
- Checkpoint saving with compression
- Timeline branching (what-if scenarios)
- Delta encoding for efficiency

## Architecture

```
AlchemicalSynth/
├── audio_preset.py      # JSON preset system
├── ca_engine.py          # CA rules (2D + hex)
├── fractal_generator.py # Fractal noise algorithms
├── terrain_synth.py     # Core audio engine
├── pattern_detector.py  # Pattern → music triggers
├── graph_persistence.py # Checkpoint/branch system
├── schemas/             # JSON schemas
└── synth_plugin/        # JUCE C++ plugin
    ├── CMakeLists.txt
    ├── Source/
    │   └── PluginProcessor.h
    └── README.md
```

## Preset JSON Schema

```json
{
  "name": "My Preset",
  "version": "1.0",
  "terrain": {
    "type": "ca_fractal_hybrid",
    "grid_size": [128, 128],
    "ca_rule": "conway",
    "fractal_type": "perlin",
    "fractal_octaves": 4,
    "fractal_dim": 1.5,
    "hybrid_blend": 0.5
  },
  "trajectory": {
    "shape": "ellipse",
    "frequency": 220,
    "meanderance": 0.2,
    "feedback": {"enabled": false}
  },
  "synthesis": {
    "waveform": "sine",
    "filter": {"type": "lowpass", "cutoff": 8000}
  },
  "envelope": {
    "attack": 0.01,
    "decay": 0.2,
    "sustain": 0.7,
    "release": 0.5
  }
}
```

## Building the JUCE Plugin

### Prerequisites
- CMake 3.15+
- JUCE framework

### Build

```bash
cd SynthLab/synth_plugin
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### Install

**macOS (AU for GarageBand):**
```bash
cp -r build/AlchemicalSynth.component ~/Library/Audio/Plug-Ins/Components/
```

**Windows (VST3):**
```bash
cp -r build/AlchemicalSynth.vst3 "C:\Program Files\Common Files\VST3\"
```

## Development Roadmap

### Phase 1: Core Audio Engine ✅
- [x] CA engine with multiple rules
- [x] Fractal generators  
- [x] Terrain synthesis
- [x] Trajectory modulation
- [x] Pattern detection

### Phase 2: JUCE Plugin ⏳
- [x] Project structure
- [ ] Full C++ implementation
- [ ] Test on GarageBand

### Phase 3: Advanced Features
- [ ] Hex-grid CA support for unique timbres
- [ ] Neural network latent injection
- [ ] Real-time CA evolution audio
- [ ] MIDI pattern triggering

### Phase 4: Generative Composition
- [ ] Graph-based composition rules
- [ ] World DNA for complete generative worlds
- [ ] Multiple timeline branching

## Research References

This implementation draws from research on:
- **Wave Terrain Synthesis** - Scanning 2D surfaces for audio
- **Cellular Automata in Music** - Chaosynth, CAMUS, PQCA
- **Fractal Audio** - 1/f noise, dimension-based timbre
- **Neural Cellular Automata** - Learned CA rules for texture

See `Hybrid Generative Soundscape Synthesis.txt` for full research overview.

## License

GPL-3.0 (same as main AlchemicalLab project)
