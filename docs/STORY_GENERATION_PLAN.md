# Exploratory Story Generation System: Planning Document

## Overview
This system enables exploratory story generation through story node graphs, leveraging local LLMs for world-building and narrative creation. It builds robust information frameworks for consistency, with a critic process to maintain coherence. The system starts with a "world-DNA" Markdown file seeding a world context, which LLMs and generative tools flesh out into interconnected stories.

## Key Components
- **World-DNA Model**: MD file defining core world elements (setting, themes, characters, conflicts). Serves as seed for generative expansion.
- **LLM Engine**: Local LLMs (e.g., Qwen2.5, DeepSeek-R1) for ideation, scaffolding, and narrative generation. GPU-accelerated via llama.cpp.
- **Critic System**: Rule-based checks + LLM critic for consistency (e.g., plot coherence, character arcs).
- **API Integrations**: External APIs for inspiration (news, books, images) to augment prompts and enhance creativity.
- **Story Node Graphs**: Dynamic graphs (nodes: elements; edges: narrative connections with depth). Stored in graph/vector DBs.
- **Web UI**: D3.js-based HUD for visualizations, user feedback, and world evaluation.
- **Evaluation**: Metrics comparing world/story depth to sci-fi/fantasy classics (e.g., LOTR, Dune).

## Architecture
- **Backend**: Python/C++ (llama.cpp integration), graph DB (Neo4j), vector DB (Chroma), API clients (requests).
- **Frontend**: Web app with D3.js for interactive graphs.
- **Workflow**:
  1. Load world-DNA.
  2. Fetch API data (optional: news/books/images).
  3. LLM generates world expansions with API inspiration.
  4. Critic validates.
  5. User feedback refines.
  6. Build story graphs.
  7. Visualize/evaluate.

## API Integrations
- **News APIs**: NewsAPI/Currents for real-time headlines to inspire plots (e.g., tech/science events).
- **Story Archives**: Gutendex (Gutenberg), Archive.org, Open Library for literary excerpts/summaries.
- **Image Generation**: Replicate (Stable Diffusion), Unsplash, Craiyon for visuals.
- **Additional**: Freesound (audio samples), gTTS (TTS), Nominatim (geocoding), PoetryDB (poems), Hugging Face (sentiment).
- **Implementation**: Modular functions in `api_integrations.py`; cached with LRU; toggles in UI; fallbacks for errors. Focus on free/unlimited APIs.

## Implementation Steps
1. Set up GPU inference (CUDA in llama.cpp).
2. Download/test LLMs.
3. Implement world-DNA parser.
4. Build critic system.
5. Develop web UI.
6. Add API integrations (news, books, archives, images, audio, etc.).
7. Add evaluation.

## Dependencies
- llama.cpp (with CUDA)
- Ollama/Python bindings
- Neo4j/Chroma
- D3.js/Streamlit
- SpaCy for rules
- requests for APIs

## Current Status
- **Core System**: World-DNA loading, LLM generation (Qwen/DeepSeek/Phi/TinyLlama), critic system (rule-based + LLM), story graphs (Chroma DB), web UI (Flask + D3.js).
- **API Integrations**: Fully implemented with caching/error handling. Includes news (NewsAPI/Currents), books (Gutendex random genres), archives (Archive.org/Open Library), images (Replicate/Craiyon), audio (Freesound/gTTS), poems (PoetryDB), sentiment (Hugging Face), geocoding (Nominatim), research (Wikipedia/NASA).
- **Enhancements**: Randomized API pulls, dynamic prompt summarization, critique loop research, UI theme inputs, batch generation.
- **Testing**: Ready for browser testing and performance profiling.

## Setup Guide
1. **Prerequisites**: Python 3.8+, CUDA-compatible GPU, llama.cpp built with CUDA.
2. **Install Dependencies**: `pip install -r requirements.txt` (includes flask, requests, chromadb, gtts, replicate, etc.).
3. **Download Models**: Place GGUF models (qwen.gguf, deepseek-r1.gguf, etc.) in project root.
4. **Set API Keys**: Optional env vars: NEWSAPI_KEY, REPLICATE_API_TOKEN, HF_TOKEN, NASA_API_KEY, FREESOUND_API_KEY.
5. **Run**: `cd StoryLab && python app.py`. Open http://127.0.0.1:5000.
6. **Test**: Generate stories with API toggles, check graphs, rate stories.

## Risks & Mitigations
- LLM hallucinations: Critic + user feedback.
- Performance: Optimize for 16GB VRAM.
- Complexity: Start simple, iterate.