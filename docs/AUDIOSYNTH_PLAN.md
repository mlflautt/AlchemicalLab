# AlchemicalSynth Development Plan

## Vision

Create a generative audio synthesis system that combines cellular automata, fractal geometry, and neural networks to produce evolving, organic soundscapes. The system should work both as a standalone Python library and as an AU/VST3 plugin for GarageBand and other DAWs.

## Current Status

### Completed ✅

| Component | File | Status |
|-----------|------|--------|
| CA Engine | `ca_engine.py` | Complete - 16 rules |
| Fractal Generator | `fractal_generator.py` | Complete - 10+ types |
| Terrain Synth | `terrain_synth.py` | Complete - Working |
| Pattern Detector | `pattern_detector.py` | Complete |
| Audio Presets | `audio_preset.py` | Complete |
| Graph Persistence | `graph_persistence.py` | Complete |
| JUCE Project | `synth_plugin/` | Structure only |

### In Progress ⏳

| Component | Status | Notes |
|-----------|--------|-------|
| C++ Plugin | Header only | Needs full implementation |
| Hex CA | Partial | Basic support exists |
| Pattern audio triggers | Basic | Needs MIDI output |

### Not Started 🚧

| Component | Priority | Notes |
|-----------|----------|-------|
| Full C++ implementation | High | Need JUCE setup |
| Real-time CA evolution | Medium | Performance optimization |
| Neural network integration | Low | If time permits |
| Graph-to-audio composition | Medium | Advanced feature |

## How It Works

### Wave Terrain Synthesis

The core concept is scanning a 2D "terrain" surface with a moving point (trajectory):

```
┌─────────────────────────────┐
│  Terrain (CA/Fractal)       │
│  ┌───────────────────────┐  │
│  │    ╱─────╮            │  │
│  │   ╱       ╲           │  │  ← Trajectory path
│  │  ╱         ╲          │  │
│  └───────────────────────┘  │
│                             │
│  Height at point → Audio   │
└─────────────────────────────┘
```

1. **Terrain Generation**: CA grid or fractal noise creates height map
2. **Trajectory**: 2D path (ellipse, lissajous, etc.) scans terrain
3. **Sampling**: Height at trajectory position modulates waveform
4. **Processing**: Filter, envelope, effects

### CA for Audio

Cellular automata produce evolving patterns. Different rules create different sonic characteristics:

| CA Rule | Audio Character |
|---------|-----------------|
| Conway | Chaotic, evolving textures |
| Brian's Brain | Pulsing, wave-like |
| Rule 30 | High-frequency chaos |
| Rule 90 | Sierpinski = rhythmic |
| Seeds | Explosive attacks |
| HighLife | Stable patterns |

### Fractal Dimension & Timbre

Research shows fractal dimension (D) correlates with perceptual "grittiness":

| D Value | Noise Type | Timbre |
|--------|------------|--------|
| 2.0 | White | Harsh, chaotic |
| 1.5 | Pink | Organic, natural |
| 1.0 | Brown | Smooth, dark |

## Technical Details

### CA Engine
- 2D grid (128x128 default, configurable)
- 16 rules including 1D rules extended to 2D
- Hexagonal grid support
- Multi-state (2-10 states)
- Pattern detection for music triggers

### Fractal Generator
- Perlin, Simplex, Worley, Voronoi
- Mandelbrot, Julia sets
- FBM (Fractal Brownian Motion)
- Ridged multifractal
- Domain warping
- Configurable octaves, lacunarity, persistence

### Audio Processing
- Sample rate: 44100 Hz (configurable)
- Bit depth: 32-bit float
- Waveforms: sine, saw, square, triangle, sample-hold
- Filter: 1-pole lowpass with resonance
- Envelope: ADSR with ES (envelope affects size) toggle

### Pattern Detection
- Connected component analysis
- Pattern classification (glider, oscillator, still life)
- Velocity estimation
- Dynamic analysis (growth/decay trends)
- Audio trigger generation

## Integration with AlchemicalLab

The audio system can use:

1. **CA from CALab**: Use existing CA systems as terrain
2. **Graph from GraphEngine**: Composition rules from knowledge graph
3. **World DNA**: Presets define world parameters
4. **Fractals from FractalLab**: Mandelbrot/Julia terrain generation

## Research Basis

This system is inspired by:

1. **Chaosynth** - CA-controlled granular synthesis
2. **CAMUS** - CA for macro-structural composition  
3. **Terrain Synth** - Wave terrain synthesis plugin
4. **RAVE** - Neural audio synthesis
5. **DyNCA** - Neural CA for dynamic textures

See `Hybrid Generative Soundscape Synthesis.txt` for full research references.

## Future Enhancements

### Short Term
1. Complete C++ plugin implementation
2. Add more CA rules (Margolus neighborhood)
3. Improve pattern detection accuracy
4. Add MIDI output to pattern detector

### Medium Term
1. Real-time CA evolution in plugin
2. Hexagonal CA for unique timbres
3. Multiple trajectory layers
4. Preset library with example sounds

### Long Term
1. Neural network latent injection
2. Graph-based composition rules
3. Full World DNA integration
4. Multiple timeline management

## Contributing

To extend the system:

1. **Add CA Rule**: Edit `ca_engine.py`, add to `RULE_DEFINITIONS`
2. **Add Fractal**: Edit `fractal_generator.py`, add method + `hybrid()` case
3. **Add Trajectory**: Edit `terrain_synth.py`, add to `TrajectoryShape` and `compute_trajectory_position()`
4. **Add Preset Type**: Edit `audio_preset_schema.json`

## Testing

```bash
# Test CA engine
python -c "from SynthLab.ca_engine import CAEngine, CARule; ca = CAEngine((50,50), CARule.CONWAY); ca.initialize_random(0.3); ca.step(); print('CA OK')"

# Test fractal
python -c "from SynthLab.fractal_generator import FractalGenerator; fg = FractalGenerator((64,64), 42); print(fg.perlin().shape)"

# Test synth
python -c "from SynthLab.terrain_synth import TerrainSynthesizer; s = TerrainSynthesizer(); t = s.generate_terrain(); print(t.shape)"

# Test pattern
python -c "from SynthLab.pattern_detector import PatternDetector; pd = PatternDetector(); print('Pattern OK')"
```

## License

GPL-3.0 - Same as AlchemicalLab main project
