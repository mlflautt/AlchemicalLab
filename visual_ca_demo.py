"""
Visual CA Sound Demo - Shows CA grids alongside audio analysis.
Run: python visual_ca_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from SynthLab.ca_engine import CAEngine, CARule
from SynthLab.terrain_synth import TerrainSynthesizer


def visualize_rules():
    """Create visualization of different CA rules and their sounds."""
    
    rules = [
        ('conway', 'Conway\'s Game of Life'),
        ('brians_brain', 'Brian\'s Brain'),
        ('day_night', 'Day & Night'),
        ('rule90', 'Rule 90 (Sierpinski)'),
    ]
    
    synth = TerrainSynthesizer(sample_rate=22050)
    
    fig, axes = plt.subplots(len(rules), 4, figsize=(16, 12))
    fig.suptitle('CA Rules and Their Sound Characteristics', fontsize=14)
    
    for idx, (rule_name, title) in enumerate(rules):
        # Create and evolve CA
        ca = CAEngine((64, 64), CARule(rule_name), seed=42)
        ca.initialize_random(0.3)
        
        # Evolve for 20 generations
        densities = []
        for _ in range(20):
            ca.step()
            densities.append(ca.get_density())
        
        # Column 1: CA Grid
        axes[idx, 0].imshow(ca.grid, cmap='binary')
        axes[idx, 0].set_title(f'{title}\nFinal Density: {densities[-1]:.3f}')
        axes[idx, 0].axis('off')
        
        # Column 2: Density over time
        axes[idx, 1].plot(densities, 'b-', linewidth=2)
        axes[idx, 1].fill_between(range(len(densities)), densities, alpha=0.3)
        axes[idx, 1].set_ylabel('Density')
        axes[idx, 1].set_ylim(0, 0.6)
        axes[idx, 1].set_title('CA Evolution')
        
        # Column 3: Generate audio from final state
        terrain = ca.get_terrain_height()
        params = {
            'shape': 'ellipse',
            'frequency': 220,
            'harmonics': [1, 2, 3],
            'harmonic_amps': [1.0, 0.5, 0.25],
            'filter': {'type': 'lowpass', 'cutoff': 6000},
            'envelope': {'attack': 0.01, 'decay': 0.1, 'sustain': 0.8, 'release': 0.3},
        }
        
        synth.trigger_attack()
        audio = synth.generate_block(terrain, params, 2048)
        
        # Column 4: Audio waveform
        axes[idx, 2].plot(audio[:500], 'g-', linewidth=0.5)
        axes[idx, 2].set_title(f'RMS: {np.sqrt(np.mean(audio**2)):.4f}')
        axes[idx, 2].set_ylim(-1, 1)
        
        # Column 5: Spectrogram (simplified)
        axes[idx, 3].specgram(audio, Fs=22050, cmap='viridis')
        axes[idx, 3].set_title('Spectrogram')
    
    plt.tight_layout()
    plt.savefig('ca_visual_sound.png', dpi=150)
    print("Saved: ca_visual_sound.png")


def show_evolution_sound():
    """Show how CA evolution affects sound over time."""
    
    ca = CAEngine((64, 64), CARule.CONWAY, seed=42)
    ca.initialize_random(0.4)
    
    synth = TerrainSynthesizer(sample_rate=22050)
    
    # Track evolution
    states = []
    rms_values = []
    
    print("Generation | Grid Density | Audio RMS")
    print("-"*45)
    
    for gen in range(20):
        terrain = ca.get_terrain_height()
        
        params = {
            'shape': 'ellipse',
            'frequency': 220,
            'harmonics': [1, 2, 3],
            'harmonic_amps': [1.0, 0.5, 0.25],
            'filter': {'type': 'lowpass', 'cutoff': 5000},
            'envelope': {'attack': 0.01, 'decay': 0.1, 'sustain': 0.8, 'release': 0.3},
        }
        
        synth.trigger_attack()
        audio = synth.generate_block(terrain, params, 512)
        
        density = np.mean(ca.grid)
        rms = np.sqrt(np.mean(audio**2))
        
        print(f"    {gen:3d}    |    {density:.3f}     |  {rms:.4f}")
        
        states.append(ca.grid.copy())
        rms_values.append(rms)
        
        ca.step()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('CA Evolution and Sound', fontsize=14)
    
    # First 4 generations
    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.imshow(states[i * 5], cmap='binary')
        ax.set_title(f'Gen {i*5}: Density={np.mean(states[i*5]):.3f}, RMS={rms_values[i*5]:.4f}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('ca_evolution_sound.png', dpi=150)
    print("\nSaved: ca_evolution_sound.png")


def compare_trajectories():
    """Visualize how different trajectories scan terrain."""
    
    synth = TerrainSynthesizer()
    terrain = synth.generate_terrain('perlin')
    
    shapes = ['ellipse', 'figure8', 'lissajous', 'rose']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Trajectory Paths and Their Audio', fontsize=14)
    
    for idx, shape in enumerate(shapes):
        # Get trajectory path
        xs, ys = [], []
        for phase in np.linspace(0, 4*np.pi, 200):
            x, y = synth.compute_trajectory_position(shape, phase, {'scale': 0.3})
            xs.append(x)
            ys.append(y)
        
        # Plot trajectory on terrain
        ax = axes[0, idx]
        ax.imshow(terrain, cmap='terrain', alpha=0.7)
        ax.plot([x*128 for x in xs], [y*128 for y in ys], 'r-', linewidth=1.5)
        ax.set_title(f'{shape.capitalize()} Trajectory')
        ax.axis('off')
        
        # Generate audio
        params = {
            'shape': shape,
            'frequency': 220,
            'harmonics': [1, 2, 3],
            'harmonic_amps': [1.0, 0.5, 0.25],
            'filter': {'type': 'lowpass', 'cutoff': 8000},
            'envelope': {'attack': 0.01, 'decay': 0.2, 'sustain': 0.7, 'release': 0.5},
        }
        
        synth.trigger_attack()
        audio = synth.generate_block(terrain, params, 1024)
        
        # Plot waveform
        ax = axes[1, idx]
        ax.plot(audio[:256], 'b-', linewidth=0.5)
        ax.set_title(f'RMS: {np.sqrt(np.mean(audio**2)):.4f}')
        ax.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150)
    print("Saved: trajectory_comparison.png")


if __name__ == '__main__':
    print("="*60)
    print("Visual CA + Sound Demonstration")
    print("="*60)
    
    visualize_rules()
    show_evolution_sound()
    compare_trajectories()
    
    print("\nAll visualizations saved as PNG files!")
