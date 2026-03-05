"""
Web Server for Real-time SynthLab Visualization

Serves a web interface that displays the multi-layer CA simulation
with live updates using D3.js for visualization.
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from SynthLab.hybrid_framework import SemanticCA

class SimulationServer:
    """Web server that runs CA simulation and serves data to web clients."""

    def __init__(self, grid_size=(50, 50), port=8080):
        self.grid_size = grid_size
        self.port = port
        self.ca = SemanticCA(grid_size=grid_size, seed=42)
        self.running = False
        self.step_count = 0
        self.stats_history = []

    def start_simulation(self):
        """Start the simulation thread."""
        self.running = True
        self.sim_thread = threading.Thread(target=self._simulation_loop)
        self.sim_thread.daemon = True
        self.sim_thread.start()

    def stop_simulation(self):
        """Stop the simulation."""
        self.running = False
        if hasattr(self, 'sim_thread'):
            self.sim_thread.join()

    def _simulation_loop(self):
        """Main simulation loop."""
        while self.running:
            # Run one step
            state = self.ca.step({}, {})

            # Collect statistics
            stats = {
                'step': self.step_count,
                'alive_count': int(np.sum(state['alive'])),
                'avg_energy': float(np.mean(state['energy'])),
                'species_diversity': int(len(np.unique(state['species']))),
                'avg_age': float(np.mean(state['age'])),
                'avg_health': float(np.mean(state['health'])),
                'avg_wealth': float(np.mean(state['wealth'])),
                'avg_tech': float(np.mean(state['technological_level'])),
                'total_resources': float(np.sum(state['resources']))
            }

            self.stats_history.append(stats)
            # Keep only last 100 steps
            if len(self.stats_history) > 100:
                self.stats_history = self.stats_history[-100:]

            self.step_count += 1
            time.sleep(0.1)  # 10 FPS

    def get_current_data(self):
        """Get current visualization data."""
        viz_data = self.ca.get_visualization_data()
        return {
            'grids': {
                'alive': viz_data['alive_grid'].tolist(),
                'species': viz_data['species_grid'].tolist(),
                'energy': viz_data['energy_grid'].tolist(),
                'color': viz_data['color_grid'].tolist(),
                'resources': viz_data['resource_grid'].tolist()
            },
            'stats': self.stats_history[-1] if self.stats_history else {},
            'grid_size': self.grid_size
        }

class RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the visualization server."""

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path == '/':
            # Serve main HTML page
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('SynthLab/web/index.html', 'rb') as f:
                self.wfile.write(f.read())

        elif parsed_path.path == '/data':
            # Serve current simulation data as JSON
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            data = self.server.sim_server.get_current_data()
            self.wfile.write(json.dumps(data).encode())

        elif parsed_path.path == '/style.css':
            # Serve CSS
            self.send_response(200)
            self.send_header('Content-type', 'text/css')
            self.end_headers()
            with open('SynthLab/web/style.css', 'rb') as f:
                self.wfile.write(f.read())

        elif parsed_path.path == '/script.js':
            # Serve JavaScript
            self.send_response(200)
            self.send_header('Content-type', 'application/javascript')
            self.end_headers()
            with open('SynthLab/web/script.js', 'rb') as f:
                self.wfile.write(f.read())

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')

    def log_message(self, format, *args):
        """Suppress log messages."""
        pass

def run_server(grid_size=(50, 50), port=8080):
    """Run the visualization server."""
    # Create simulation server
    sim_server = SimulationServer(grid_size=grid_size, port=port)
    sim_server.start_simulation()

    # Create HTTP server
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    httpd.sim_server = sim_server

    print(f"AlchemicalLab Visualization Server running on http://localhost:{port}")
    print("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        sim_server.stop_simulation()
        httpd.shutdown()

if __name__ == "__main__":
    run_server()