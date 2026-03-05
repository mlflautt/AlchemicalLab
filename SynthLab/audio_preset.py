"""
Audio Preset Loader for AlchemicalSynth.

Loads and manages audio preset JSON files for the terrain synthesizer.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class AudioPreset:
    """Audio preset configuration."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self._validate()
    
    def _validate(self):
        """Validate preset data against schema."""
        required = ['name', 'version', 'terrain', 'trajectory']
        for field in required:
            if field not in self.data:
                raise ValueError(f"Missing required field: {field}")
        
        if 'type' not in self.data['terrain']:
            raise ValueError("Missing terrain.type")
        
        if 'shape' not in self.data['trajectory']:
            raise ValueError("Missing trajectory.shape")
        
        if 'frequency' not in self.data['trajectory']:
            raise ValueError("Missing trajectory.frequency")
    
    @property
    def name(self) -> str:
        return self.data.get('name', 'Untitled')
    
    @property
    def version(self) -> str:
        return self.data.get('version', '1.0')
    
    @property
    def terrain_type(self) -> str:
        return self.data['terrain']['type']
    
    @property
    def grid_size(self):
        return self.data['terrain'].get('grid_size', [128, 128])
    
    @property
    def ca_rule(self) -> str:
        return self.data['terrain'].get('ca_rule', 'conway')
    
    @property
    def ca_params(self) -> Dict:
        return self.data['terrain'].get('ca_params', {})
    
    @property
    def fractal_type(self) -> str:
        return self.data['terrain'].get('fractal_type', 'perlin')
    
    @property
    def fractal_params(self) -> Dict:
        return {
            'octaves': self.data['terrain'].get('fractal_octaves', 4),
            'lacunarity': self.data['terrain'].get('fractal_lacunarity', 2.0),
            'persistence': self.data['terrain'].get('fractal_persistence', 0.5),
            'fractal_dim': self.data['terrain'].get('fractal_dim', 1.5),
        }
    
    @property
    def hybrid_blend(self) -> float:
        return self.data['terrain'].get('hybrid_blend', 0.5)
    
    @property
    def seed(self) -> Optional[int]:
        return self.data['terrain'].get('seed')
    
    @property
    def trajectory_shape(self) -> str:
        return self.data['trajectory']['shape']
    
    @property
    def frequency(self) -> float:
        return self.data['trajectory']['frequency']
    
    @property
    def harmonics(self):
        return self.data['trajectory'].get('harmonics', [1, 2, 3, 5])
    
    @property
    def meanderance(self) -> float:
        return self.data['trajectory'].get('meanderance', 0.2)
    
    @property
    def meander_speed(self) -> float:
        return self.data['trajectory'].get('meander_speed', 0.1)
    
    @property
    def feedback_enabled(self) -> bool:
        return self.data['trajectory'].get('feedback', {}).get('enabled', False)
    
    @property
    def feedback_params(self) -> Dict:
        return self.data['trajectory'].get('feedback', {
            'delay_samples': 4410,
            'compression': 0.8,
            'radius': 1.0
        })
    
    @property
    def envelope(self) -> Dict:
        return self.data.get('envelope', {
            'attack': 0.01,
            'decay': 0.2,
            'sustain': 0.7,
            'release': 0.5
        })
    
    @property
    def synthesis_params(self) -> Dict:
        return self.data.get('synthesis', {
            'waveform': 'sine',
            'oversampling': 4,
            'filter': {'type': 'lowpass', 'cutoff': 8000, 'resonance': 0.5},
            'saturation': 0
        })
    
    def to_dict(self) -> Dict:
        return self.data.copy()
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.data, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AudioPreset':
        return cls(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AudioPreset':
        return cls(json.loads(json_str))
    
    @classmethod
    def from_file(cls, path: str) -> 'AudioPreset':
        with open(path, 'r') as f:
            return cls(json.load(f))
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)


class PresetLibrary:
    """Manages a collection of audio presets."""
    
    def __init__(self, library_path: str = None):
        self.library_path = library_path
        self.presets: Dict[str, AudioPreset] = {}
        
        if library_path and os.path.exists(library_path):
            self.load_library()
    
    def load_library(self):
        """Load all presets from library directory."""
        if not self.library_path:
            return
        
        for root, dirs, files in os.walk(self.library_path):
            for file in files:
                if file.endswith('.json'):
                    try:
                        path = os.path.join(root, file)
                        preset = AudioPreset.from_file(path)
                        self.presets[preset.name] = preset
                    except Exception as e:
                        print(f"Warning: Failed to load {file}: {e}")
    
    def add_preset(self, preset: AudioPreset):
        """Add a preset to the library."""
        self.presets[preset.name] = preset
    
    def get_preset(self, name: str) -> Optional[AudioPreset]:
        return self.presets.get(name)
    
    def list_presets(self) -> list:
        return list(self.presets.keys())
    
    def save_preset(self, preset: AudioPreset, filename: str = None):
        """Save a preset to the library."""
        if not self.library_path:
            return
        
        if filename is None:
            filename = f"{preset.name.replace(' ', '_')}.json"
        
        path = os.path.join(self.library_path, filename)
        preset.save(path)


def create_default_preset() -> AudioPreset:
    """Create a default preset with sensible defaults."""
    data = {
        "name": "Default Terrain",
        "version": "1.0",
        "description": "Basic wave terrain synthesis preset",
        "terrain": {
            "type": "ca_fractal_hybrid",
            "grid_size": [128, 128],
            "ca_rule": "conway",
            "ca_params": {
                "initial_density": 0.3,
                "evolution_speed": 0.1,
                "states": 2,
                "neighborhood": "moore"
            },
            "fractal_type": "perlin",
            "fractal_octaves": 4,
            "fractal_lacunarity": 2.0,
            "fractal_persistence": 0.5,
            "fractal_dim": 1.5,
            "hybrid_blend": 0.5,
            "seed": 42
        },
        "trajectory": {
            "shape": "ellipse",
            "frequency": 220,
            "harmonics": [1, 2, 3, 5],
            "meanderance": 0.2,
            "meander_speed": 0.1,
            "feedback": {
                "enabled": False,
                "delay_samples": 4410,
                "compression": 0.8,
                "radius": 1.0
            },
            "translation": {
                "x_speed": 0,
                "y_speed": 0,
                "circular": False
            }
        },
        "synthesis": {
            "waveform": "sine",
            "oversampling": 4,
            "filter": {
                "type": "lowpass",
                "cutoff": 8000,
                "resonance": 0.5
            },
            "saturation": 0
        },
        "envelope": {
            "attack": 0.01,
            "decay": 0.2,
            "sustain": 0.7,
            "release": 0.5,
            "es_toggle": False
        },
        "modulation": {
            "lfo_frequency": 1,
            "lfo_depth": 0.3,
            "lfo_target": "frequency"
        },
        "spatial": {
            "stereo_width": 1,
            "pan_mode": "stereo"
        }
    }
    return AudioPreset(data)


if __name__ == '__main__':
    preset = create_default_preset()
    print(f"Created preset: {preset.name}")
    print(preset.to_json())
