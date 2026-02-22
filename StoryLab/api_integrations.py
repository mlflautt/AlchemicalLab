#!/usr/bin/env python3
"""
API Integrations for AlchemicalLab: Fetch external data for story inspiration.
Includes news, books, images with caching and error handling.
"""

import requests
import os
from functools import lru_cache
from typing import List, Optional
import io
import random

# API Keys (set as env vars for security)
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')  # Optional for NewsAPI
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN', '')  # For image gen

@lru_cache(maxsize=10)
def fetch_news(api: str = 'currents', query: str = 'technology', limit: int = 2) -> List[str]:
    """Fetch news headlines for inspiration."""
    try:
        if api == 'newsapi' and NEWSAPI_KEY:
            url = f"https://newsapi.org/v2/everything?q={query}&pageSize={limit}&apiKey={NEWSAPI_KEY}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [article['title'] for article in data.get('articles', [])[:limit]]
        elif api == 'currents':
            url = f"https://api.currentsapi.services/v1/search?keywords={query}&language=en&limit={limit}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [news['title'] for news in data.get('news', [])[:limit]]
    except Exception as e:
        print(f"News API error: {e}")
    return ["No news available"]

@lru_cache(maxsize=5)
def fetch_book_excerpt(query: str = 'science fiction', author: str = '', title: str = '', limit: int = 1, random_genre: bool = False) -> str:
    """Fetch excerpt from Project Gutenberg via Gutendex. If random_genre=True, pick random genre."""
    try:
        # If random_genre, override query with random genre
        if random_genre:
            genres = ['science fiction', 'fantasy', 'mystery', 'romance', 'horror', 'adventure', 'historical fiction', 'biography', 'drama', 'comedy']
            query = random.choice(genres)

        # Build search URL
        search_terms = []
        if query:
            search_terms.append(query)
        if author:
            search_terms.append(f"author:{author}")
        if title:
            search_terms.append(f"title:{title}")
        search_query = ' '.join(search_terms)
        search_url = f"https://gutendex.com/books?search={search_query.replace(' ', '%20')}"
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                # Pick random book if multiple results
                book = random.choice(data['results'])
                text_url = book['formats'].get('text/plain')
                if text_url:
                    text_response = requests.get(text_url, timeout=10)
                    if text_response.status_code == 200:
                        text = text_response.text
                        # Extract first 500 chars after *** START
                        start = text.find('*** START')
                        if start != -1:
                            excerpt = text[start + 10:start + 510]
                            return excerpt.replace('\n', ' ').strip()
    except Exception as e:
        print(f"Gutenberg API error: {e}")
    return "No book excerpt available"

@lru_cache(maxsize=5)
def fetch_wiki_summary(topic: str = 'mythology') -> str:
    """Fetch Wikipedia summary."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('extract', 'No summary available')[:300]
    except Exception as e:
        print(f"Wikipedia API error: {e}")
    return "No summary available"

@lru_cache(maxsize=5)
def generate_image(prompt: str = 'fractured planet landscape') -> str:
    """Generate image via Replicate (Stable Diffusion) or Craiyon."""
    try:
        if REPLICATE_API_TOKEN:
            try:
                import replicate
                replicate.Client(api_token=REPLICATE_API_TOKEN)
                output = replicate.run(
                    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                    input={"prompt": prompt, "num_inference_steps": 20}
                )
                return output[0] if output else "No image generated"
            except ImportError:
                print("Replicate not installed, skipping Replicate API")
    except Exception as e:
        print(f"Replicate API error: {e}")
    # Fallback to Craiyon (free AI art)
    try:
        response = requests.post("https://api.craiyon.com/v1/generate", json={"prompt": prompt}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data['images'][0] if data['images'] else "No image generated"
    except Exception as e:
        print(f"Craiyon API error: {e}")
    # Final fallback to Unsplash
    try:
        url = f"https://api.unsplash.com/search/photos?query={prompt.replace(' ', '%20')}&per_page=1"
        headers = {"Authorization": "Client-ID YOUR_UNSPLASH_ACCESS_KEY"}  # Replace with key if available
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                return data['results'][0]['urls']['small']
    except Exception as e:
        print(f"Unsplash API error: {e}")
    return "https://via.placeholder.com/300x200?text=No+Image"

@lru_cache(maxsize=5)
def text_to_speech(text: str, lang: str = 'en') -> str:
    """Generate TTS audio via gTTS (free, unlimited)."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text[:500], lang=lang, slow=False)  # Limit text
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        # Save to file or return data URL
        filename = f"tts_{hash(text)}.mp3"
        with open(filename, 'wb') as f:
            f.write(audio_buffer.getvalue())
        return filename  # Local path
    except ImportError:
        print("gTTS not installed, TTS unavailable")
        return "No audio generated (gTTS not installed)"
    except Exception as e:
        print(f"TTS error: {e}")
    return "No audio generated"

@lru_cache(maxsize=5)
def geocode_location(location: str) -> dict:
    """Geocode location via Nominatim (free, rate limited)."""
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location.replace(' ', '%20')}&format=json&limit=1"
        response = requests.get(url, headers={"User-Agent": "StoryLab/1.0"}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                return {"lat": data[0]['lat'], "lon": data[0]['lon'], "display_name": data[0]['display_name']}
    except Exception as e:
        print(f"Geocode error: {e}")
    return {"lat": "40.7128", "lon": "-74.0060", "display_name": "Default NYC"}

@lru_cache(maxsize=5)
def fetch_poem(author: str = '', title: str = '') -> str:
    """Fetch poem from PoetryDB (free)."""
    try:
        if author:
            url = f"https://poetrydb.org/author/{author}/title"
        elif title:
            url = f"https://poetrydb.org/title/{title}"
        else:
            url = "https://poetrydb.org/random/1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            poems = response.json()
            if poems:
                poem = poems[0]
                return f"{poem['title']} by {poem['author']}\n\n" + "\n".join(poem['lines'][:20])  # Limit lines
    except Exception as e:
        print(f"PoetryDB error: {e}")
    return "No poem available"

@lru_cache(maxsize=5)
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment via Hugging Face free API."""
    try:
        url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}  # Use token if available
        response = requests.post(url, headers=headers, json={"inputs": text[:512]}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data[0], list):
                labels = {item['label']: item['score'] for item in data[0]}
                return labels
    except Exception as e:
        print(f"Sentiment error: {e}")
    return {"neutral": 1.0}  # Fallback

@lru_cache(maxsize=5)
def fetch_audio_sample(query: str = 'ambient') -> str:
    """Fetch audio sample from Freesound (free, requires API key)."""
    try:
        api_key = os.getenv('FREESOUND_API_KEY', '')
        if api_key:
            url = f"https://freesound.org/apiv2/search/text/?query={query}&token={api_key}&fields=id,name,previews"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    return data['results'][0]['previews']['preview-lq-mp3']
    except Exception as e:
        print(f"Freesound error: {e}")
    return "No audio sample"

@lru_cache(maxsize=5)
def fetch_archive_text(query: str = 'fantasy stories', limit: int = 1) -> str:
    """Fetch text excerpt from Archive.org (free)."""
    try:
        search_url = f"https://archive.org/advancedsearch.php?q={query.replace(' ', '%20')}&fl[]=identifier,title&sort[]=downloads%20desc&rows={limit}&output=json"
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'response' in data and 'docs' in data['response']:
                for doc in data['response']['docs'][:limit]:
                    item_id = doc['identifier']
                    meta_url = f"https://archive.org/metadata/{item_id}"
                    meta_response = requests.get(meta_url, timeout=10)
                    if meta_response.status_code == 200:
                        meta = meta_response.json()
                        if 'files' in meta:
                            for file_key, file_info in meta['files'].items():
                                if file_key.endswith('.txt') and 'format' in file_info and file_info['format'] == 'Text':
                                    text_url = f"https://archive.org/download/{item_id}/{file_key}"
                                    text_response = requests.get(text_url, timeout=10)
                                    if text_response.status_code == 200:
                                        text = text_response.text
                                        excerpt = text[:500].replace('\n', ' ').strip()
                                        return f"From {doc.get('title', 'Unknown')}: {excerpt}"
    except Exception as e:
        print(f"Archive.org error: {e}")
    return "No archival text available"

@lru_cache(maxsize=5)
def fetch_open_library_excerpt(title: str = 'dune') -> str:
    """Fetch book excerpt from Open Library (free)."""
    try:
        search_url = f"https://openlibrary.org/search.json?title={title.replace(' ', '%20')}&limit=1"
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'docs' in data and data['docs']:
                book = data['docs'][0]
                key = book.get('key')
                if key:
                    details_url = f"https://openlibrary.org{key}.json"
                    details_response = requests.get(details_url, timeout=10)
                    if details_response.status_code == 200:
                        details = details_response.json()
                        description = details.get('description', '')
                        if isinstance(description, str):
                            return description[:300]
                        elif isinstance(description, dict) and 'value' in description:
                            return description['value'][:300]
    except Exception as e:
        print(f"Open Library error: {e}")
    return "No excerpt available"

@lru_cache(maxsize=5)
def research_topic_wikipedia(topic: str) -> str:
    """Research topic via Wikipedia API (free)."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('extract', 'No summary')[:500]
    except Exception as e:
        print(f"Wikipedia research error: {e}")
    return "No research data"

@lru_cache(maxsize=5)
def research_science_nasa(query: str = 'black holes') -> str:
    """Research science via NASA API (free, requires key)."""
    try:
        api_key = os.getenv('NASA_API_KEY', 'DEMO_KEY')  # DEMO_KEY for testing
        url = f"https://api.nasa.gov/planetary/apod?api_key={api_key}&concept_tags=True"
        # For simplicity, fetch APOD; extend for search
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return f"NASA Insight: {data.get('explanation', '')[:300]}"
    except Exception as e:
        print(f"NASA research error: {e}")
    return "No science data"

@lru_cache(maxsize=5)
def research_biome_wikipedia(biome: str = 'temperate forest') -> str:
    """Research biome for world-building via Wikipedia."""
    return research_topic_wikipedia(biome)

@lru_cache(maxsize=5)
def research_historical_event_wikipedia(event: str = 'industrial revolution') -> str:
    """Research historical event for depth."""
    return research_topic_wikipedia(event)

@lru_cache(maxsize=5)
def fetch_folklore(country: str = 'random') -> str:
    """Fetch folklore from a random country or specified."""
    if country == 'random':
        countries = ['Japan', 'India', 'Germany', 'Brazil', 'Egypt', 'Russia', 'China', 'Mexico', 'Greece', 'Ireland']
        country = random.choice(countries)
    topic = f'folklore of {country}'
    return research_topic_wikipedia(topic)

if __name__ == "__main__":
    # Test
    print("News:", fetch_news())
    print("Book:", fetch_book_excerpt())
    print("Wiki:", fetch_wiki_summary())
    print("Image:", generate_image())