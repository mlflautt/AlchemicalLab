"""
Quick CA Sound Examples - Demonstrates how different CA rules produce different audio.
Run: python quick_ca_examples.py
"""

import numpy as np
from SynthLab.ca_engine import CAEngine, CARule
from SynthLab.terrain_synth import TerrainSynthesizer
from SynthLab.pattern_detector import DynamicPatternDetector, PatternDetector


def quick_ca_demo():
    """Quick demonstration of CA-driven audio."""
    
    print("="*60)
    print("CA RULE SOUND COMPARISON")
    print("="*60)
    
    rules = [
        ('conway', 'Chaotic, evolving patterns'),
        ('brians_brain', 'Wave-like, pulsing'), 
        ('highlife', 'Complex, interesting'),
        ('day_night', 'Structured symmetry'),
        ('seeds', 'Explosive growth'),
        ('rule90', 'Sierpinski rhythm'),
    ]
    
    synth = TerrainSynthesizer(sample_rate=22050)
    results = []
    
    for rule_name, desc in rules:
        print(f"\n>>> {rule_name}: {desc}")
        
        # Create CA
        ca = CAEngine((32, 32), CARule(rule_name), seed=42)
        ca.initialize_random(0.3)
        
        # Quick evolve
        for _ in range(10):
            ca.step()
        
        terrain = ca.get_terrain_height()
        
        # Generate short audio
        traj_params = {
            'shape': 'ellipse',
            'frequency': 220,
            'harmonics': [1, 2, 3],
            'harmonic_amps': [1.0, 0.5, 0.25],
            'radius_x': 0.35,
            'radius_y': 0.35,
            'filter': {'type': 'lowpass', 'cutoff': 6000},
            'envelope': {'attack': 0.01, 'decay': 0.1, 'sustain': 0.8, 'release': 0.2},
        }
        
        synth.trigger_attack()
        audio = synth.generate_block(terrain, traj_params, 1024)
        
        # Analyze
        rms = np.sqrt(np.mean(audio**2))
        density = np.mean(ca.grid)
        
        print(f"    RMS: {rms:.4f}  |  CA Density: {density:.3f}")
        
        results.append({
            'rule': rule_name,
            'description': desc,
            'rms': rms,
            'density': density,
            'audio': audio
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Rule':<15} {'Description':<30} {'RMS':>8} {'Density':>8}")
    print("-"*65)
    for r in results:
        print(f"{r['rule']:<15} {r['description']:<30} {r['rms']:>8.4f} {r['density']:>8.3f}")
    
    return results


def pattern_demo():
    """Show pattern detection in action."""
    print("\n" + "="*60)
    print("PATTERN DETECTION → MUSIC TRIGGERS")
    print("="*60)
    
    ca = CAEngine((40, 40), CARule.CONWAY, seed=42)
    ca.initialize_random(0.2)
    
    detector = DynamicPatternDetector()
    
    print("\nGen | Density | Activity | Patterns")
    print("-"*50)
    
    for gen in range(20):
        patterns = detector.detect(ca.grid)
        dynamics = detector.analyze_dynamics(ca.grid)
        ca.step()
        
        ptypes = set(p.pattern_type.value for p in patterns) if patterns else {}
        
        print(f" {gen:3d} | {dynamics['density']:.3f}  | {dynamics['activity']:.3f}  | {ptypes if ptypes else 'none'}")
    
    # Get final triggers
    triggers = detector.get_audio_triggers(detector.detected_patterns)
    print(f"\nMusic triggers from patterns:")
    print(f"  Note ON events:  {len(triggers['note_on'])}")
    print(f"  Rhythm events:  {len(triggers['rhythm'])}")


def trajectory_demo():
    """Show trajectory shape effects."""
    print("\n" + "="*60)
    print("TRAJECTORY SHAPE EFFECTS")
    print("="*60)
    
    synth = TerrainSynthesizer()
    terrain = synth.generate_terrain('perlin')
    
    shapes = ['ellipse', 'figure8', 'lissajous', 'rose']
    
    print(f"\nShape          | RMS Energy | Peak    | Character")
    print("-"*60)
    
    for shape in shapes:
        params = {
            'shape': shape,
            'frequency': 220,
            'harmonics': [1, 2, 3],
            'harmonic_amps': [1.0, 0.5, 0.25],
            'radius_x': 0.3,
            'radius_y': 0.3,
            'filter': {'type': 'lowpass', 'cutoff': 8000},
            'envelope': {'attack': 0.01, 'decay': 0.2, 'sustain': 0.7, 'release': 0.5},
        }
        
        synth.trigger_attack()
        
        try:
            audio = synth.generate_block(terrain, params, 2048)
            
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            
            # Describe character
            if shape == 'ellipse':
                char = "smooth, continuous"
            elif shape == 'figure8':
                char = "figure-8 pattern"
            elif shape == 'lissajous':
                char = "complex harmonic"
            elif shape == 'rose':
                char = "petal-like rhythm"
            else:
                char = "random, evolving"
            
            print(f"{shape:<14} | {rms:9.4f} | {peak:7.4f} | {char}")
        except Exception as e:
            print(f"{shape:<14} | Error: {e}")


def ca_evolution_demo():
    """Show how CA evolves over time affects sound."""
    print("\n" + "="*60)
    print("CA EVOLUTION OVER TIME")
    print("="*60)
    
    ca = CAEngine((50, 50), CARule.CONWAY, seed=42)
    ca.initialize_random(0.4)
    
    synth = TerrainSynthesizer(sample_rate=22050)
    
    print("\nGen | CA Density | Audio RMS | Notes")
    print("-"*45)
    
    for gen in range(15):
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
        
        rms = np.sqrt(np.mean(audio**2))
        density = np.mean(ca.grid)
        
        # Pattern count
        pd = PatternDetector()
        patterns = pd.detect(ca.grid)
        
        note_count = len(pd.get_audio_triggers(patterns)['note_on'])
        
        print(f" {gen:3d} | {density:.3f}      | {rms:.4f}   | {note_count} notes")
        
        ca.step()


if __name__ == '__main__':
    quick_ca_demo()
    pattern_demo()
    trajectory_demo()
    ca_evolution_demo()
    
    print("\n" + "="*60)
    print("Complete! These demos show how CA drives audio:")
    print("1. Different CA rules = different sound textures")
    print("2. Pattern detection triggers musical events")  
    print("3. Trajectory shapes affect harmonic content")
    print("4. CA evolution creates changing soundscapes")
    print("="*60)
