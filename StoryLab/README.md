# StoryLab: Exploratory Generative Narratives

StoryLab is an interactive system for generating and exploring emergent narratives using AI, built on the AlchemicalLab framework. It combines large language models (LLMs) with vector databases, graph analysis, and web-based visualization to create interconnected story worlds.

## Features

- **Multi-Step Story Generation**: Hierarchical generation from world-DNA to detailed plots.
- **AI Critic System**: Dual-layer validation for coherence and quality.
- **Vector & Graph Databases**: Chroma for semantic search, NetworkX for relationship mapping.
- **Interactive Web UI**: D3.js visualizations with controls for exploration.
- **Evaluation Metrics**: Quantitative assessment of complexity and alignment.
- **Model Flexibility**: Support for DeepSeek-R1 and Qwen models.

## Setup

1. **Prerequisites**:
   - Python 3.10+
   - llama.cpp built with CUDA (for GPU acceleration)
   - Models: Download `deepseek-1.gguf`, `qwen_q2.gguf`, `tinyllama.gguf` to root directory.

2. **Install Dependencies**:
   ```bash
   pip install -r ../requirements.txt
   pip install chromadb sentence-transformers psutil
   ```

3. **Run**:
   ```bash
   cd StoryLab
   python app.py
   ```
   Open http://127.0.0.1:5000 in browser.

## Usage

- **Generate Stories**: Select model, toggle multi-step, click "Generate".
- **Explore Graphs**: View story connections; zoom, filter entities, search.
- **Edit World**: Modify DNA for custom worlds.
- **Batch Mode**: Generate multiple stories at once.
- **Export**: Download graph data as JSON.

## Architecture

- `story_generator.py`: Core generation with critic.
- `db_manager.py`: Vector/graph storage and entity extraction.
- `metrics.py`: Evaluation functions.
- `app.py`: Flask backend.
- `templates/index.html`: D3.js frontend.

## Testing

Run automated tests:
```bash
python test_runner.py
```

Check `generation_log.txt` and `test_results.json` for metrics.

## Future Plans

- Full CA integration for world evolution.
- Advanced entity relationships.
- Multi-modal outputs (images, audio).

## License

Part of AlchemicalLab - experimental AI research.