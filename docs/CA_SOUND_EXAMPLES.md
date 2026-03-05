# CA Sound Output Examples

This document shows example outputs demonstrating how different Cellular Automata rules drive audio synthesis.

---

## 1. CA Rule Sound Comparison

Different CA rules produce distinctly different audio characteristics:

| Rule | Description | RMS Energy | CA Density | Sound Character |
|------|-------------|------------|-------------|------------------|
| **conway** | Chaotic, evolving | 0.2476 | 0.234 | Evolving, unpredictable |
| **brians_brain** | Wave-like, pulsing | 0.3009 | 0.510 | Rhythmic, pulsing |
| **highlife** | Complex, interesting | 0.2562 | 0.263 | Rich, layered |
| **day_night** | Structured symmetry | 0.2940 | 0.368 | Balanced, periodic |
| **seeds** | Explosive growth | 0.2361 | 0.200 | Sparse, dynamic |
| **rule90** | Sierpinski rhythm | 0.2933 | 0.449 | Rhythmic, patterned |

### Observations:
- **Brian's Brain** produces the highest energy (RMS: 0.3009) - 3-state rule creates more activity
- **Seeds** produces lowest energy - explosive but sparse patterns
- **Rule 90** creates regular Sierpinski patterns - good for rhythmic textures

---

## 2. Pattern Detection → Music Triggers

The pattern detector identifies CA patterns and generates musical events:

```
Gen | Density | Activity | Patterns
--------------------------------------------------
   0 | 0.214  | 0.000  | {oscillator, still_life, unknown}
   1 | 0.221  | 0.000  | {glider, oscillator, still_life, decay}
   2 | 0.189  | 0.000  | {oscillator, still_life, decay, glider}
   3 | 0.198  | 0.000  | {spaceship, oscillator, still_life, glider}
```

**Music triggers generated:**
- Note ON events: 14
- Rhythm events: 1

### Pattern → Music Mapping:
| CA Pattern | Audio Trigger |
|-------------|----------------|
| Glider moving | Melody note |
| Oscillator | Rhythm (kick/snare) |
| Still life | Drone/ambient |
| Decay | Descending pitch |
| Spaceship | Moving melody |

---

## 3. Trajectory Shape Effects

Different trajectory shapes scan the terrain differently, affecting the sound:

| Shape | RMS Energy | Peak | Character |
|-------|------------|------|-----------|
| ellipse | 0.4634 | 0.8389 | smooth, continuous |
| figure8 | 0.4975 | 1.0000 | figure-8 pattern |
| lissajous | 0.4822 | 0.8440 | complex harmonic |
| rose | 0.4841 | 1.0000 | petal-like rhythm |

### Observations:
- **Figure-8** produces highest energy - crosses terrain more
- **Ellipse** is smoothest - predictable path
- **Rose** creates rhythmic variation - periodic distance changes

---

## 4. CA Evolution Over Time

As the CA evolves, both the grid density and audio output change:

```
Gen | CA Density | Audio RMS | Notes
---------------------------------------------
   0 | 0.401      | 0.2225   | Dense initial state
   1 | 0.352      | 0.2689   | 
   2 | 0.317      | 0.2592   | 
   5 | 0.272      | 0.2552   | 
  10 | 0.206      | 0.2415   | 
  14 | 0.176      | 0.2480   | Stabilized
```

### Observations:
- CA density decreases from 0.401 → 0.176 over 15 generations
- Audio RMS stays relatively stable (0.22-0.27) despite grid changes
- System stabilizes after ~10 generations

---

## How to Run Examples

```bash
# Quick demo showing all features
python quick_ca_examples.py

# Full demo (slower, saves audio files)
python ca_sound_examples.py
```

---

## Key Findings

1. **CA Rule Choice Matters**: Different rules produce distinctly different timbres
2. **Pattern Detection Works**: Can identify gliders, oscillators, still life
3. **Trajectory Affects Sound**: Shape determines scanning pattern
4. **Evolution Creates Movement**: CA dynamics create evolving textures
5. **Density Doesn't Predict Loudness**: Low density rules can still produce rich audio

---

## Audio Generation Pipeline

```
CA Grid (64x64) → Height at (x,y) → Oscillator → Filter → Envelope → Output
      ↑                                      ↑
   Evolves                               Trajectory
   over time                             scans terrain
```

The trajectory position (x, y) determines which height from the terrain grid is sampled, which then controls the oscillator waveform.
