from flask import Flask, render_template, request, jsonify
import os
import sys
import logging
import time
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from story_generator import load_world_dna, generate_story_idea
from db_manager import StoryDBManager
from metrics import StoryMetrics

logging.basicConfig(filename='generation_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

app = Flask(__name__)
print("DEBUG: Initializing DB and metrics...")
try:
    db = StoryDBManager()
    metrics = StoryMetrics()
    print("DEBUG: Metrics initialized successfully.")
except Exception as e:
    print(f"DEBUG: Failed to initialize metrics: {e}")
    db = None
    metrics = None

@app.route('/')
def index():
    dna = load_world_dna("world_dna.md")
    return render_template('index.html', dna=dna)

@app.route('/generate', methods=['POST'])
def generate():
    start_time = time.time()
    data = request.get_json() or {}
    multi_step = data.get('multi_step', False)
    model = data.get('model', 'qwen')
    theme = data.get('theme', '')
    use_news = data.get('use_news', False)
    use_books = data.get('use_books', False)
    use_random_books = data.get('use_random_books', False)
    use_archive = data.get('use_archive', False)
    use_open_library = data.get('use_open_library', False)
    use_images = data.get('use_images', False)
    use_tts = data.get('use_tts', False)
    use_poem = data.get('use_poem', False)
    use_sentiment = data.get('use_sentiment', False)
    if model == 'deepseek':
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deepseek-r1.gguf'))
    elif model == 'phi3':
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phi3-mini.gguf'))
    elif model == 'tinyllama':
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tinyllama-reasoning.gguf'))
    else:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'qwen_q2.gguf'))
    print(f"DEBUG: Starting generation with model={model}, multi_step={multi_step}")
    dna = load_world_dna("world_dna.md")
    print(f"DEBUG: Loaded DNA, length={len(dna)}")
    try:
        print("DEBUG: Calling generate_story_idea...")
        story, critique, extra = generate_story_idea(dna, model_path=model_path, db=db, multi_step=multi_step, theme=theme, use_news=use_news, use_books=use_books, use_random_books=use_random_books, use_archive=use_archive, use_open_library=use_open_library, use_images=use_images, use_tts=use_tts, use_poem=use_poem, use_sentiment=use_sentiment)
        print(f"DEBUG: Generation completed, story length={len(story)}")
        logging.info(f"Generation completed: score={critique['overall_score']:.2f}")
        eval_metrics = {}
        if db:
            print("DEBUG: Building connections...")
            db.build_connections()  # Update connections
        if metrics:
            print("DEBUG: Evaluating story...")
            eval_metrics = metrics.evaluate_story(story, dna, db.graph if db else None)
        end_time = time.time()
        logging.info(f"Full process: time={end_time-start_time:.2f}s, score={critique['overall_score']:.2f}")
        print(f"DEBUG: Returning result, total time={end_time-start_time:.2f}s")
        return jsonify({'story': story, 'critique': critique, 'metrics': eval_metrics, 'extra': extra})
    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        logging.error(f"Generation failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/graph')
def get_graph():
    if db:
        return jsonify(db.get_graph_data())
    return jsonify({"nodes": [], "edges": []})

@app.route('/rate', methods=['POST'])
def rate():
    data = request.get_json()
    rating = data.get('rating', 5)
    logging.info(f"User rating: {rating}")
    return jsonify({'status': 'ok'})

@app.route('/update_dna', methods=['POST'])
def update_dna():
    data = request.get_json()
    new_dna = data.get('dna', '')
    with open('world_dna.md', 'w') as f:
        f.write(new_dna)
    logging.info("DNA updated")
    return jsonify({'status': 'ok'})

@app.route('/evolve_dna', methods=['POST'])
def evolve_dna():
    # Placeholder: Use metrics to adjust DNA parameters
    dna = load_world_dna("world_dna.md")
    # Simple: Increase art emphasis if coherence high
    if db and db.graph.number_of_nodes() > 10:
        dna = dna.replace("Art Emphasis: High (0.8)", "Art Emphasis: Very High (0.9)")
    with open('world_dna.md', 'w') as f:
        f.write(dna)
    logging.info("DNA evolved based on stories")
    return jsonify({'status': 'ok', 'new_dna': dna})

@app.route('/batch_generate', methods=['POST'])
def batch_generate():
    data = request.get_json() or {}
    num = data.get('num', 5)
    multi_step = data.get('multi_step', False)
    model = data.get('model', 'qwen')
    theme = data.get('theme', '')
    use_news = data.get('use_news', False)
    use_books = data.get('use_books', False)
    use_random_books = data.get('use_random_books', False)
    use_archive = data.get('use_archive', False)
    use_open_library = data.get('use_open_library', False)
    use_images = data.get('use_images', False)
    use_tts = data.get('use_tts', False)
    use_poem = data.get('use_poem', False)
    use_sentiment = data.get('use_sentiment', False)
    if model == 'deepseek':
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deepseek-r1.gguf'))
    elif model == 'phi3':
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phi3-mini.gguf'))
    elif model == 'tinyllama':
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tinyllama-reasoning.gguf'))
    else:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'qwen_q2.gguf'))
    dna = load_world_dna("world_dna.md")
    results = []
    for i in range(num):
        story, critique, extra = generate_story_idea(dna, model_path=model_path, db=db, multi_step=multi_step, theme=theme, use_news=use_news, use_books=use_books, use_random_books=use_random_books, use_archive=use_archive, use_open_library=use_open_library, use_images=use_images, use_tts=use_tts, use_poem=use_poem, use_sentiment=use_sentiment)
        eval_metrics = {}
        if metrics and db:
            eval_metrics = metrics.evaluate_story(story, dna, db.graph)
        results.append({'story': story, 'critique': critique, 'metrics': eval_metrics, 'extra': extra})
        logging.info(f"Batch {i}: score {critique['overall_score']}")
    if db:
        db.build_connections()
    return jsonify(results)

@app.route('/clear_db', methods=['POST'])
def clear_db():
    global db
    if db:
        db.collection.delete(where={})
        db.graph.clear()
    logging.info("DB cleared")
    return jsonify({'status': 'ok'})

# --- CALab Integration ---
from CALab.narrative_emergence import WorldBuildingSystem
import numpy as np
import json

class NumpySetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.route('/calab/generate', methods=['POST'])
def calab_generate():
    data = request.get_json() or {}
    generations = data.get('generations', 100)
    seed = data.get('seed', 42)
    
    print(f"DEBUG: Starting CALab generation for {generations} generations with seed {seed}...")
    
    try:
        world = WorldBuildingSystem(world_size=(100, 100))
        world.initialize_world(density=0.3, seed=seed)
        world.run_headless(generations)
        world_data = world.get_world_data()
        
        print("DEBUG: CALab generation complete. Sending data.")
        return json.dumps(world_data, indent=2, cls=NumpySetEncoder)
        
    except Exception as e:
        print(f"DEBUG: CALab Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        logging.error(f"CALab Generation failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- FractalLab Integration ---
from FractalLab.generators.mandelbrot import generate_mandelbrot
from flask import send_file
import io

@app.route('/fractal/generate', methods=['POST'])
def fractal_generate():
    data = request.get_json() or {}
    width = data.get('width', 800)
    height = data.get('height', 800)
    x_center = data.get('x_center', -0.75)
    y_center = data.get('y_center', 0)
    zoom = data.get('zoom', 1)
    max_iter = data.get('max_iter', 256)

    print(f"DEBUG: Starting FractalLab generation with zoom {zoom}...")

    try:
        img = generate_mandelbrot(width, height, x_center, y_center, zoom, max_iter)
        
        # Save image to a byte stream
        byte_io = io.BytesIO()
        img.save(byte_io, 'PNG')
        byte_io.seek(0)
        
        print("DEBUG: FractalLab generation complete. Sending image.")
        return send_file(byte_io, mimetype='image/png')

    except Exception as e:
        print(f"DEBUG: FractalLab Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        logging.error(f"FractalLab Generation failed: {str(e)}")
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    print("DEBUG: Starting Flask app...")
    from werkzeug.serving import make_server
    server = make_server('127.0.0.1', 5000, app, threaded=True)
    print("Server created, starting...")
    server.serve_forever()