"""
AlchemicalSynth Web GUI

A web-based interface for exploring CA-driven sound synthesis.
Shows CA grid, trajectory path, terrain, and audio in real-time.

Run: python app.py
Then open http://localhost:5000
"""

from flask import Flask, render_template_string, jsonify, request, send_file
import io
import base64
import numpy as np
import threading
import time
import soundfile as sf
import tempfile
import os

app = Flask(__name__)

# Import SynthLab modules
from SynthLab.ca_engine import CAEngine, CARule
from SynthLab.fractal_generator import FractalGenerator
from SynthLab.terrain_synth import TerrainSynthesizer
from SynthLab.pattern_detector import PatternDetector


class SynthState:
    """Global synthesizer state shared across requests."""
    
    def __init__(self):
        self.grid_size = 64
        self.sample_rate = 22050
        
        # CA state
        self.ca = None
        self.ca_rule = 'conway'
        self.ca_seed = 42
        self.ca_generation = 0
        
        # Terrain
        self.terrain = None
        self.fractal_type = 'perlin'
        
        # Synthesizer
        self.synth = None
        self.trajectory_shape = 'ellipse'
        self.frequency = 220
        self.hybrid_blend = 0.5
        
        # Audio
        self.current_audio = None
        self.audio_lock = threading.Lock()
        
        # Pattern detector
        self.pattern_detector = PatternDetector()
        
        # Running state
        self.running = False
        self.lock = threading.Lock()
        
        self._init()
    
    def _init(self):
        """Initialize all components."""
        # Create CA
        try:
            ca_rule = CARule(self.ca_rule)
        except ValueError:
            ca_rule = CARule.CONWAY
        
        self.ca = CAEngine((self.grid_size, self.grid_size), ca_rule, seed=self.ca_seed)
        self.ca.initialize_random(0.3)
        
        # Create synth
        self.synth = TerrainSynthesizer(sample_rate=self.sample_rate)
        self.synth.grid_size = (self.grid_size, self.grid_size)
        self.synth.hybrid_blend = self.hybrid_blend
        self.synth._init_default_terrain()
        
        # Generate initial terrain
        self._update_terrain()
    
    def _update_terrain(self):
        """Update terrain based on current settings."""
        if self.synth:
            self.terrain = self.synth.generate_terrain(self.fractal_type)
    
    def step(self):
        """Step the simulation forward."""
        with self.lock:
            if self.ca:
                self.ca.step()
                self.ca_generation += 1
                self.terrain = self.ca.get_terrain_height()
    
    def generate_audio(self, duration=2.0):
        """Generate audio block."""
        with self.audio_lock:
            if self.synth is None or self.terrain is None:
                return None
            
            traj_params = {
                'shape': self.trajectory_shape,
                'frequency': self.frequency,
                'harmonics': [1, 2, 3, 5],
                'harmonic_amps': [1.0, 0.5, 0.25, 0.125],
                'radius_x': 0.35,
                'radius_y': 0.35,
                'filter': {'type': 'lowpass', 'cutoff': 6000},
                'envelope': {'attack': 0.01, 'decay': 0.1, 'sustain': 0.8, 'release': 0.3},
            }
            
            samples = int(duration * self.sample_rate)
            self.synth.trigger_attack()
            audio = self.synth.generate_block(self.terrain, traj_params, samples)
            
            return audio
    
    def get_trajectory_path(self, num_points=100):
        """Get trajectory path coordinates."""
        if self.synth is None:
            return [], []
        
        xs, ys = [], []
        phases = np.linspace(0, 4 * np.pi, num_points)
        
        for phase in phases:
            x, y = self.synth.compute_trajectory_position(
                self.trajectory_shape, 
                phase, 
                {'scale': 0.3}
            )
            xs.append(x)
            ys.append(y)
        
        return xs, ys
    
    def get_patterns(self):
        """Get detected patterns."""
        if self.ca is None:
            return []
        
        patterns = self.pattern_detector.detect(self.ca.grid)
        
        return [
            {
                'type': p.pattern_type.value,
                'center': p.center,
                'confidence': p.confidence
            }
            for p in patterns
        ]
    
    def set_ca_rule(self, rule_name):
        """Change CA rule."""
        with self.lock:
            try:
                ca_rule = CARule(rule_name)
                self.ca = CAEngine(
                    (self.grid_size, self.grid_size), 
                    ca_rule, 
                    seed=self.ca_seed
                )
                self.ca.initialize_random(0.3)
                self.ca_generation = 0
                self.ca_rule = rule_name
                self._update_terrain()
            except ValueError:
                pass
    
    def set_trajectory(self, shape):
        """Change trajectory shape."""
        self.trajectory_shape = shape
    
    def set_frequency(self, freq):
        """Change frequency."""
        self.frequency = freq


# Global state
state = SynthState()


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlchemicalSynth - CA Sound Explorer</title>
    <script src="https://cdn.jsdelivr.net/npm/p5.js@1.9.0/lib/p5.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
        }
        .header {
            background: #161b22;
            padding: 15px 20px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 1.4rem;
            color: #58a6ff;
        }
        .header .tagline {
            color: #8b949e;
            font-size: 0.9rem;
        }
        .main {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        .panel {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
        }
        .panel h2 {
            font-size: 1rem;
            color: #58a6ff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #30363d;
        }
        #ca-canvas, #terrain-canvas {
            width: 100%;
            border-radius: 4px;
            display: block;
        }
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 0.85rem;
        }
        .stat {
            background: #0d1117;
            padding: 8px;
            border-radius: 4px;
        }
        .stat-label { color: #8b949e; }
        .stat-value { color: #58a6ff; font-weight: bold; }
        .controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .control-group {
            background: #0d1117;
            padding: 12px;
            border-radius: 6px;
        }
        .control-group label {
            display: block;
            color: #8b949e;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }
        select, input[type="range"] {
            width: 100%;
            background: #161b22;
            border: 1px solid #30363d;
            color: #c9d1d9;
            padding: 8px;
            border-radius: 4px;
        }
        select:focus, input:focus {
            outline: none;
            border-color: #58a6ff;
        }
        .buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        button {
            background: #238636;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.2s;
        }
        button:hover { background: #2ea043; }
        button.secondary { background: #30363d; }
        button.secondary:hover { background: #484f58; }
        button.danger { background: #da3633; }
        button.danger:hover { background: #f85149; }
        .audio-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        #play-btn.playing {
            background: #f0883e;
        }
        audio {
            width: 100%;
            margin-top: 10px;
        }
        .patterns {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 10px;
        }
        .pattern-tag {
            background: #1f6feb;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
        }
        .info {
            font-size: 0.8rem;
            color: #8b949e;
            margin-top: 10px;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>AlchemicalSynth</h1>
            <span class="tagline">Cellular Automata → Sound Synthesis</span>
        </div>
    </div>
    
    <div class="main">
        <!-- Left Panel: CA Grid -->
        <div class="panel">
            <h2>🧬 Cellular Automaton</h2>
            <canvas id="ca-canvas"></canvas>
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Generation</div>
                    <div class="stat-value" id="generation">0</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Density</div>
                    <div class="stat-value" id="density">0.00</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Active Cells</div>
                    <div class="stat-value" id="active-cells">0</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Rule</div>
                    <div class="stat-value" id="rule-name">conway</div>
                </div>
            </div>
            <div style="margin-top: 15px;">
                <strong>Detected Patterns:</strong>
                <div class="patterns" id="patterns"></div>
            </div>
        </div>
        
        <!-- Center Panel: Terrain + Trajectory -->
        <div class="panel">
            <h2>⛰️ Terrain + Trajectory</h2>
            <canvas id="terrain-canvas"></canvas>
            <div class="info">
                The trajectory (red line) scans the terrain surface.<br>
                Height at trajectory position → Audio output.
            </div>
        </div>
        
        <!-- Right Panel: Controls + Audio -->
        <div class="panel">
            <h2>🎛️ Controls</h2>
            <div class="controls">
                <div class="control-group">
                    <label>CA Rule</label>
                    <select id="ca-rule">
                        <option value="conway">Conway's Game of Life</option>
                        <option value="brians_brain">Brian's Brain</option>
                        <option value="highlife">HighLife</option>
                        <option value="day_night">Day & Night</option>
                        <option value="seeds">Seeds</option>
                        <option value="rule90">Rule 90 (Sierpinski)</option>
                        <option value="rule30">Rule 30 (Chaotic)</option>
                        <option value="replicator">Replicator</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Trajectory Shape</label>
                    <select id="trajectory-shape">
                        <option value="ellipse">Ellipse</option>
                        <option value="figure8">Figure-8</option>
                        <option value="lissajous">Lissajous</option>
                        <option value="rose">Rose Curve</option>
                        <option value="spiral">Spiral</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Frequency: <span id="freq-value">220</span> Hz</label>
                    <input type="range" id="frequency" min="60" max="880" value="220">
                </div>
                
                <div class="control-group">
                    <label>Fractal Type</label>
                    <select id="fractal-type">
                        <option value="perlin">Perlin Noise</option>
                        <option value="worley">Worley (Cellular)</option>
                        <option value="ridged">Ridged</option>
                        <option value="fbm">FBM</option>
                    </select>
                </div>
                
                <div class="buttons">
                    <button id="step-btn">Step</button>
                    <button id="play-btn">▶ Play</button>
                    <button id="generate-btn">🔊 Generate Audio</button>
                </div>
                
                <div class="control-group">
                    <label>Generated Audio</label>
                    <audio id="audio-player" controls></audio>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Canvas contexts
        let caCanvas, caCtx;
        let terrainCanvas, terrainCtx;
        let caData = null;
        let terrainData = null;
        let trajectoryData = null;
        let isPlaying = false;
        
        // Initialize
        function init() {
            caCanvas = document.getElementById('ca-canvas');
            caCtx = caCanvas.getContext('2d');
            terrainCanvas = document.getElementById('terrain-canvas');
            terrainCtx = terrainCanvas.getContext('2d');
            
            // Set canvas sizes
            caCanvas.width = caCanvas.offsetWidth;
            caCanvas.height = caCanvas.offsetWidth;
            terrainCanvas.width = terrainCanvas.offsetWidth;
            terrainCanvas.height = terrainCanvas.offsetWidth;
            
            // Event listeners
            document.getElementById('ca-rule').addEventListener('change', changeCARule);
            document.getElementById('trajectory-shape').addEventListener('change', changeTrajectory);
            document.getElementById('frequency').addEventListener('input', changeFrequency);
            document.getElementById('fractal-type').addEventListener('change', changeFractal);
            document.getElementById('step-btn').addEventListener('click', stepCA);
            document.getElementById('play-btn').addEventListener('click', togglePlay);
            document.getElementById('generate-btn').addEventListener('click', generateAudio);
            
            // Start update loop
            updateLoop();
        }
        
        function updateLoop() {
            fetchData();
            drawCA();
            drawTerrain();
            
            if (isPlaying) {
                stepCA();
            }
            
            setTimeout(updateLoop, isPlaying ? 100 : 500);
        }
        
        function fetchData() {
            fetch('/api/state')
                .then(r => r.json())
                .then(data => {
                    caData = data.ca_grid;
                    terrainData = data.terrain;
                    trajectoryData = data.trajectory;
                    
                    document.getElementById('generation').textContent = data.generation;
                    document.getElementById('density').textContent = data.density.toFixed(3);
                    document.getElementById('active-cells').textContent = data.active_cells;
                    document.getElementById('rule-name').textContent = data.rule;
                    
                    // Update patterns
                    const patternDiv = document.getElementById('patterns');
                    patternDiv.innerHTML = data.patterns.map(p => 
                        `<span class="pattern-tag">${p.type}</span>`
                    ).join('');
                });
        }
        
        function drawCA() {
            if (!caData || !caCtx) return;
            
            const size = caCanvas.width;
            const gridSize = caData.length;
            const cellSize = size / gridSize;
            
            const imageData = caCtx.createImageData(size, size);
            
            for (let y = 0; y < gridSize; y++) {
                for (let x = 0; x < gridSize; x++) {
                    const val = caData[y][x];
                    const idx = (y * cellSize * size + x * cellSize) * 4;
                    
                    // Scale up
                    for (let sy = 0; sy < cellSize; sy++) {
                        for (let sx = 0; sx < cellSize; sx++) {
                            const px = Math.floor(x * cellSize + sx);
                            const py = Math.floor(y * cellSize + sy);
                            const pIdx = (py * size + px) * 4;
                            
                            if (val > 0) {
                                imageData.data[pIdx] = 88;     // R
                                imageData.data[pIdx + 1] = 166; // G
                                imageData.data[pIdx + 2] = 255; // B
                                imageData.data[pIdx + 3] = 255; // A
                            } else {
                                imageData.data[pIdx] = 22;     // R
                                imageData.data[pIdx + 1] = 27;  // G
                                imageData.data[pIdx + 2] = 34;  // B
                                imageData.data[pIdx + 3] = 255; // A
                            }
                        }
                    }
                }
            }
            
            caCtx.putImageData(imageData, 0, 0);
        }
        
        function drawTerrain() {
            if (!terrainData || !terrainCtx) return;
            
            const size = terrainCanvas.width;
            const gridSize = terrainData.length;
            const cellSize = size / gridSize;
            
            const imageData = terrainCtx.createImageData(size, size);
            
            for (let y = 0; y < gridSize; y++) {
                for (let x = 0; x < gridSize; x++) {
                    const val = terrainData[y][x];
                    const normalized = Math.min(1, Math.max(0, val / 10));
                    
                    // Color map: dark blue to green to yellow
                    const r = Math.floor(normalized * 255);
                    const g = Math.floor(normalized * 200);
                    const b = Math.floor((1 - normalized) * 100 + 50);
                    
                    for (let sy = 0; sy < cellSize; sy++) {
                        for (let sx = 0; sx < cellSize; sx++) {
                            const px = Math.floor(x * cellSize + sx);
                            const py = Math.floor(y * cellSize + sy);
                            const pIdx = (py * size + px) * 4;
                            
                            imageData.data[pIdx] = r;
                            imageData.data[pIdx + 1] = g;
                            imageData.data[pIdx + 2] = b;
                            imageData.data[pIdx + 3] = 255;
                        }
                    }
                }
            }
            
            terrainCtx.putImageData(imageData, 0, 0);
            
            // Draw trajectory overlay
            if (trajectoryData && trajectoryData.xs) {
                terrainCtx.strokeStyle = '#ff4444';
                terrainCtx.lineWidth = 2;
                terrainCtx.beginPath();
                
                trajectoryData.xs.forEach((x, i) => {
                    const px = x * size;
                    const py = trajectoryData.ys[i] * size;
                    if (i === 0) {
                        terrainCtx.moveTo(px, py);
                    } else {
                        terrainCtx.lineTo(px, py);
                    }
                });
                
                terrainCtx.stroke();
                
                // Draw current position
                const cx = trajectoryData.xs[trajectoryData.xs.length - 1] * size;
                const cy = trajectoryData.ys[trajectoryData.ys.length - 1] * size;
                terrainCtx.fillStyle = '#ff0000';
                terrainCtx.beginPath();
                terrainCtx.arc(cx, cy, 5, 0, Math.PI * 2);
                terrainCtx.fill();
            }
        }
        
        function stepCA() {
            fetch('/api/step', {method: 'POST'});
        }
        
        function changeCARule() {
            const rule = document.getElementById('ca-rule').value;
            fetch('/api/set_rule/' + rule, {method: 'POST'});
        }
        
        function changeTrajectory() {
            const shape = document.getElementById('trajectory-shape').value;
            fetch('/api/set_trajectory/' + shape, {method: 'POST'});
        }
        
        function changeFrequency() {
            const freq = document.getElementById('frequency').value;
            document.getElementById('freq-value').textContent = freq;
            fetch('/api/set_frequency/' + freq, {method: 'POST'});
        }
        
        function changeFractal() {
            const fractal = document.getElementById('fractal-type').value;
            fetch('/api/set_fractal/' + fractal, {method: 'POST'});
            setTimeout(stepCA, 100);
        }
        
        function togglePlay() {
            isPlaying = !isPlaying;
            const btn = document.getElementById('play-btn');
            btn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
            btn.classList.toggle('playing', isPlaying);
        }
        
        function generateAudio() {
            document.getElementById('generate-btn').textContent = '⏳ Generating...';
            
            fetch('/api/generate_audio', {method: 'POST'})
                .then(r => r.blob())
                .then(blob => {
                    const url = URL.createObjectURL(blob);
                    const audio = document.getElementById('audio-player');
                    audio.src = url;
                    audio.play();
                    document.getElementById('generate-btn').textContent = '🔊 Generate Audio';
                });
        }
        
        // Start
        window.onload = init;
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/state')
def get_state():
    """Get current simulation state."""
    with state.lock:
        ca_grid = state.ca.grid.tolist() if state.ca else [[]]
        density = float(np.mean(state.ca.grid)) if state.ca else 0
        active_cells = int(np.sum(state.ca.grid)) if state.ca else 0
        terrain = state.terrain.tolist() if state.terrain is not None else [[]]
        xs, ys = state.get_trajectory_path()
        patterns = state.get_patterns()
    
    return jsonify({
        'ca_grid': ca_grid,
        'terrain': terrain,
        'trajectory': {'xs': xs, 'ys': ys},
        'generation': state.ca_generation,
        'density': density,
        'active_cells': active_cells,
        'rule': state.ca_rule,
        'patterns': patterns
    })


@app.route('/api/step', methods=['POST'])
def step():
    """Step the CA forward."""
    state.step()
    return jsonify({'status': 'ok'})


@app.route('/api/set_rule/<rule>', methods=['POST'])
def set_rule(rule):
    """Change CA rule."""
    state.set_ca_rule(rule)
    return jsonify({'status': 'ok', 'rule': rule})


@app.route('/api/set_trajectory/<shape>', methods=['POST'])
def set_trajectory(shape):
    """Change trajectory shape."""
    state.set_trajectory(shape)
    return jsonify({'status': 'ok', 'shape': shape})


@app.route('/api/set_frequency/<int:freq>', methods=['POST'])
def set_frequency(freq):
    """Change frequency."""
    state.set_frequency(freq)
    return jsonify({'status': 'ok', 'frequency': freq})


@app.route('/api/set_fractal/<fractal>', methods=['POST'])
def set_fractal(fractal):
    """Change fractal type."""
    state.fractal_type = fractal
    state._update_terrain()
    return jsonify({'status': 'ok', 'fractal': fractal})


@app.route('/api/generate_audio', methods=['POST'])
def generate_audio():
    """Generate audio file."""
    audio = state.generate_audio(duration=3.0)
    
    if audio is None:
        return 'No audio', 500
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, state.sample_rate)
        temp_path = f.name
    
    # Read and return
    with open(temp_path, 'rb') as f:
        audio_data = f.read()
    
    os.unlink(temp_path)
    
    return send_file(
        io.BytesIO(audio_data),
        mimetype='audio/wav',
        as_attachment=False,
        download_name='alchemical_audio.wav'
    )


if __name__ == '__main__':
    print("="*60)
    print("AlchemicalSynth GUI")
    print("="*60)
    print("Open your browser to: http://localhost:5000")
    print("="*60)
    app.run(debug=True, port=5000)
