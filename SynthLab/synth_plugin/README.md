# AlchemicalSynth - AU/VST3 Plugin

Wave Terrain Synthesis synthesizer plugin for macOS (AU) and Windows/Linux (VST3).

## Building

### Prerequisites
- CMake 3.15+
- JUCE framework (clone as submodule or system installation)

### Build Instructions

```bash
cd SynthLab/synth_plugin
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### macOS (AU)
After building, copy the `.component` bundle to:
```
~/Library/Audio/Plug-Ins/Components/
```

### Windows (VST3)
Copy the `.vst3` bundle to:
```
C:\Program Files\Common Files\VST3\
```

## Architecture

- **PluginProcessor**: Main audio processing engine
- **TerrainEngine**: CA + Fractal terrain generation
- **CAEngine**: Cellular automata rules
- **FractalGenerator**: Fractal noise algorithms
- **Trajectory**: 2D path scanning
- **PresetManager**: JSON preset loading

## Integration with AlchemicalLab

This plugin uses the same algorithms as the Python `SynthLab/` module:
- Same CA rules (Conway, Brian's Brain, etc.)
- Same fractal types (Perlin, Worley, Mandelbrot)
- Same trajectory shapes (ellipse, lissajous, etc.)
- JSON presets compatible with Python version

## Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Frequency | 20-2000 Hz | Base pitch |
| Terrain Type | CA/Fractal/Hybrid | Terrain generation method |
| CA Rule | Conway/Brian's Brain/etc. | CA evolution rule |
| Fractal Type | Perlin/Worley/etc. | Fractal noise algorithm |
| Trajectory Shape | Ellipse/Figure8/Lissajous | Scanning path |
| Hybrid Blend | 0-1 | CA/Fractal balance |
| Filter Cutoff | 100-10000 Hz | Lowpass filter |
| Attack/Release | 0-10s | Amplitude envelope |
