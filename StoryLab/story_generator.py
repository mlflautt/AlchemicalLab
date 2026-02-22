#!/usr/bin/env python3
"""
Story Generator: Loads world-DNA and generates initial narrative ideas using LLM via llama.cpp on GPU.
Includes critic system for validation and refinement.
"""

import subprocess
import json
import time
import requests
from functools import lru_cache
from critic import combined_critic
from db_manager import StoryDBManager
import random
from api_integrations import fetch_news, fetch_book_excerpt, fetch_wiki_summary, generate_image, text_to_speech, geocode_location, fetch_poem, analyze_sentiment, fetch_audio_sample, fetch_archive_text, fetch_open_library_excerpt, research_topic_wikipedia, research_science_nasa, research_biome_wikipedia, research_historical_event_wikipedia, fetch_folklore

def load_world_dna(file_path):
    with open(file_path, 'r') as f:
        return f.read()

@lru_cache(maxsize=1)
def flesh_dna(dna_text, model_path="../qwen_q2.gguf", ngl=35):
    """Step 1: Flesh out world-DNA with LLM for detailed description."""
    prompt = f"""<|im_start|>system
You are a world-building expert.<|im_end|>
<|im_start|>user
Expand this world-DNA into a cohesive, detailed world description. Include setting, factions, themes, and key elements. Ensure logical consistency.

World-DNA:
{dna_text}
<|im_end|>
<|im_start|>assistant
"""
    cmd = [f'../llama.cpp/build/bin/llama-simple', '-m', model_path, '-ngl', str(ngl), '-n', '1024', prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    fleshed = result.stdout.strip()
    return fleshed if fleshed else dna_text  # Fallback to original

def check_dna_consistency(dna_text):
    """Basic consistency check for DNA."""
    issues = []
    if "desert" in dna_text.lower() and "arctic" in dna_text.lower():
        issues.append("Biome conflict: Desert and arctic climates incompatible.")
    if "technology" in dna_text.lower() and "no technology" in dna_text.lower():
        issues.append("Tech level conflict.")
    return issues

@lru_cache(maxsize=1)
def generate_biomes(world_dna, model_path="../qwen_q2.gguf", ngl=35):
    """Step 2: Generate detailed biome descriptions using research and LLM."""
    # Parse biomes from DNA
    import re
    biomes_match = re.search(r'Primary Biomes: (.+)', world_dna, re.IGNORECASE)
    biomes = biomes_match.group(1).split(',') if biomes_match else ['temperate forest']

    research = ""
    for biome in biomes:
        biome = biome.strip()
        research += research_biome_wikipedia(biome) + "\n"

    # LLM flesh
    prompt = f"""<|im_start|>system
You are a world-building expert.<|im_end|>
<|im_start|>user
Based on this world-DNA and research, flesh out detailed descriptions for each biome, including flora, fauna, climate, and how they fit the world.

World-DNA:
{world_dna[:500]}...

Research:
{research[:500]}
<|im_end|>
<|im_start|>assistant
"""
    cmd = [f'../llama.cpp/build/bin/llama-simple', '-m', model_path, '-ngl', str(ngl), '-n', '512', prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    fleshed_biomes = result.stdout.strip()
    return fleshed_biomes if fleshed_biomes else "No biome details available"

@lru_cache(maxsize=1)
def generate_characters(world_dna, biome_details, model_path="../qwen_q2.gguf", ngl=35):
    """Step 3: Develop characters and factions with LLM, including diverse folklore."""
    folklore = fetch_folklore('random')  # Random country folklore
    prompt = f"""<|im_start|>system
You are a character designer.<|im_end|>
<|im_start|>user
Based on world-DNA, biomes, and folklore inspiration, create 3-5 key characters and develop factions. Include backgrounds, motivations, and relationships. Draw inspiration from the folklore.

World-DNA:
{world_dna[:400]}...

Biome Details:
{biome_details[:200]}...

Folklore Inspiration:
{summarize_text(folklore, 200)}
<|im_end|>
<|im_start|>assistant
"""
    cmd = [f'../llama.cpp/build/bin/llama-simple', '-m', model_path, '-ngl', str(ngl), '-n', '512', prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    characters = result.stdout.strip()
    return characters if characters else "No characters developed"

@lru_cache(maxsize=1)
def generate_plot_arcs(world_dna, biome_details, character_details, model_path="../qwen_q2.gguf", ngl=35):
    """Step 4: Structure plot arcs and conflicts with timeline."""
    prompt = f"""<|im_start|>system
You are a plot architect.<|im_end|>
<|im_start|>user
Based on world-DNA, biomes, and characters, outline 3-5 plot arcs with conflicts, ensuring timeline consistency.

World-DNA:
{world_dna[:300]}...

Biome Details:
{biome_details[:200]}...

Character Details:
{character_details[:200]}
<|im_end|>
<|im_start|>assistant
"""
    cmd = [f'../llama.cpp/build/bin/llama-simple', '-m', model_path, '-ngl', str(ngl), '-n', '512', prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    arcs = result.stdout.strip()
    return arcs if arcs else "No plot arcs developed"

@lru_cache(maxsize=1)
def generate_historical_evolution(world_dna, character_details, plot_arcs, model_path="../qwen_q2.gguf", ngl=35, steps=5):
    """Step 5: Simulate historical evolution in steps, evolving factions over time."""
    evolution = ""
    current_state = f"Initial state: {character_details[:300]}"
    for step in range(1, steps + 1):
        prompt = f"""<|im_start|>system
You are a historian.<|im_end|>
<|im_start|>user
Based on world-DNA, current faction state, and plot arcs, simulate one step of historical evolution. Describe key events, faction changes, and conflicts.

World-DNA:
{world_dna[:200]}...

Current State (Step {step-1}):
{current_state[:300]}...

Plot Arcs:
{plot_arcs[:200]}
<|im_end|>
<|im_start|>assistant
"""
        cmd = [f'../llama.cpp/build/bin/llama-simple', '-m', model_path, '-ngl', str(ngl), '-n', '256', prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        step_evolution = result.stdout.strip()
        evolution += f"Step {step}: {step_evolution}\n\n"
        current_state = step_evolution  # Update for next step
    return evolution if evolution else "No historical evolution"

@lru_cache(maxsize=1)
def get_weather(lat=40.7128, lon=-74.0060):  # Default NYC, can be randomized or based on DNA
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            weather = data['current_weather']
            return f"Temperature: {weather['temperature']}°C, Wind: {weather['windspeed']} km/h"
        return "Weather data unavailable"
    except:
        return "Weather API error"

def translate_text(text, target_lang='es'):  # Default to Spanish
    try:
        url = "https://translate.argosopentech.com/translate"
        payload = {
            "q": text[:500],  # Limit for API
            "source": "en",
            "target": target_lang
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()['translatedText']
        return text  # Fallback to original
    except:
        return text

def summarize_text(text: str, max_len: int = 100) -> str:
    """Simple summarization: truncate or take first sentence."""
    if len(text) <= max_len:
        return text
    sentences = text.split('.')
    if sentences and sentences[0]:
        summary = sentences[0].strip() + '.'
        if len(summary) <= max_len:
            return summary
    return text[:max_len].strip() + '...' if len(text) > max_len else text

def parse_weak_areas(feedback: str) -> list:
    """Parse feedback for weak areas like science, history."""
    keywords = ['science', 'history', 'mythology', 'technology', 'biology', 'geography', 'culture', 'biome', 'event']
    areas = [k for k in keywords if k in feedback.lower()]
    return areas

def generate_multi_step_story(world_dna, model_path, ngl, max_retries, db):
    """Generate story in hierarchical steps: DNA -> arcs -> subplots -> details."""
    print("Step 1: Generating plot arcs...")
    # Step 1: High-level plot arcs
    prompt1 = f"""<|im_start|>system
Master storyteller.<|im_end|>
<|im_start|>user
World-DNA: {world_dna[:300]}...

Outline 3-5 plot arcs, numbered.
<|im_end|>
<|im_start|>assistant
"""
    cmd = ['../llama.cpp/build/bin/llama-simple', '-m', model_path, '-ngl', str(ngl), '-n', '512', prompt1]
    result1 = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    arcs = result1.stdout.strip()
    print("Arcs generated.")

    print("Step 2: Developing subplots...")
    # Step 2: Develop one arc into subplots
    prompt2 = f"""<|im_start|>system
Master storyteller.<|im_end|>
<|im_start|>user
World-DNA: {world_dna[:300]}...

Develop the first arc into subplots with characters.
<|im_end|>
<|im_start|>assistant
"""
    cmd = ['../llama.cpp/build/bin/llama-simple', '-m', model_path, '-ngl', str(ngl), '-n', '1024', prompt2]
    result2 = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    story = result2.stdout.strip()
    print("Subplots developed.")

    print("Step 3: Critiquing and refining...")
    # Critic and refine
    critique = combined_critic(story, world_dna)
    if critique['overall_score'] < 7:
        print("Refining story...")
        # Simple refinement
        prompt3 = f"""<|im_start|>system
You are a master storyteller.<|im_end|>
<|im_start|>user
Improve this story based on feedback.

Original:
{story}

Feedback: {critique['rule_issues'][:100]}... {critique['llm_feedback'][:100]}...

World-DNA:
{world_dna}

Output improved version.
<|im_end|>
<|im_start|>assistant
"""
        cmd = ['../llama.cpp/build/bin/llama-simple', '-m', model_path, '-ngl', str(ngl), '-n', '1024', prompt3]
        result3 = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        story = result3.stdout.strip()
        critique = combined_critic(story, world_dna)
        print("Story refined.")

    extra = {}
    if db:
        story_id = f"story_multi_{int(time.time())}"
        db.add_story(story_id, story, {"score": critique['overall_score'], "dna": world_dna[:50], "multi_step": True, **extra})
    print("Multi-step generation complete.")
    return story, critique, extra

def generate_story_idea(world_dna, model_path="../qwen_q2.gguf", ngl=35, max_retries=3, db=None, multi_step=False, theme='', use_news=False, use_books=False, use_random_books=False, use_archive=False, use_open_library=False, use_images=False, use_tts=False, use_poem=False, use_sentiment=False):
    # Step 1: Flesh DNA
    fleshed_dna = flesh_dna(world_dna, model_path, ngl)
    consistency_issues = check_dna_consistency(fleshed_dna)
    if consistency_issues:
        print(f"DNA Consistency Issues: {consistency_issues}")
    world_dna = fleshed_dna  # Use fleshed

    # Step 2: Generate Biomes
    biome_details = generate_biomes(world_dna, model_path, ngl)

    # Step 3: Generate Characters
    character_details = generate_characters(world_dna, biome_details, model_path, ngl)

    # Step 4: Generate Plot Arcs
    plot_arcs = generate_plot_arcs(world_dna, biome_details, character_details, model_path, ngl)

    # Step 5: Historical Evolution
    historical_evolution = generate_historical_evolution(world_dna, character_details, plot_arcs, model_path, ngl)

    if multi_step:
        return generate_multi_step_story(world_dna, model_path, ngl, max_retries, db)
    
    weather_info = get_weather()
    api_inspiration = f"\nBiome Details: {summarize_text(biome_details)}\nCharacter Details: {summarize_text(character_details)}\nPlot Arcs: {summarize_text(plot_arcs)}"
    if use_news:
        news_data = fetch_news()
        summarized_news = summarize_text(', '.join(news_data))
        api_inspiration += f"\nRecent news inspiration: {summarized_news}"
    if use_books:
        book_data = fetch_book_excerpt(random_genre=use_random_books)
        summarized_book = summarize_text(book_data)
        api_inspiration += f"\nLiterary inspiration: {summarized_book}"
    if use_archive:
        archive_data = fetch_archive_text()
        summarized_archive = summarize_text(archive_data)
        api_inspiration += f"\nArchival inspiration: {summarized_archive}"
    if use_open_library:
        ol_data = fetch_open_library_excerpt()
        summarized_ol = summarize_text(ol_data)
        api_inspiration += f"\nOpen Library inspiration: {summarized_ol}"
    if use_poem:
        poem_data = fetch_poem()
        summarized_poem = summarize_text(poem_data)
        api_inspiration += f"\nPoetic inspiration: {summarized_poem}"
    theme_inspiration = f"\nFocus on theme: {theme}" if theme else ""
    prompt = f"""<|im_start|>system
Creative story writer.<|im_end|>
<|im_start|>user
World-DNA: {world_dna[:500]}...

Current weather in the world: {weather_info}{api_inspiration}{theme_inspiration}

Generate:
1. Main plot
2. Characters
3. Subplots
4. Conflicts
<|im_end|>
<|im_start|>assistant
"""
    story = ""
    critique = {}
    for attempt in range(max_retries):
        print(f"Attempt {attempt+1}: Generating story...")
        # Generate
        cmd = ['../llama.cpp/build/bin/llama-simple', '-m', model_path, '-ngl', str(ngl), '-n', '1024', prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        story = result.stdout.strip()
        print("Story generated. Critiquing...")

        # Critic
        critique = combined_critic(story, world_dna)
        print(f"Critique score: {critique['overall_score']}")
        if critique['overall_score'] >= 7:
            print("Story accepted.")
            if db:
                story_id = f"story_{int(time.time())}_{attempt}"
                db.add_story(story_id, story, {"score": critique['overall_score'], "dna": world_dna[:50]})
            return story, critique

        # Refine prompt with feedback and research
        if attempt < max_retries - 1:
            print("Refining prompt...")
            weak_areas = parse_weak_areas(critique['llm_feedback'])
            research = ""
            for area in weak_areas:
                if 'science' in area.lower():
                    research += research_science_nasa()
                elif 'historical' in area.lower() or 'event' in area.lower():
                    research += research_historical_event_wikipedia(area)
                elif 'biome' in area.lower():
                    research += research_biome_wikipedia(area)
                else:
                    research += research_topic_wikipedia(area)
            summarized_research = summarize_text(research, 300)
            feedback = f"Previous attempt issues: {critique['rule_issues']}. LLM feedback: {critique['llm_feedback'][:200]}... Research on weak areas: {summarized_research}"
            prompt += f"\n\nImprove based on this feedback: {feedback}"

    print("Max retries reached. Returning best story.")
    extra = {
        'fleshed_dna': world_dna,
        'biome_details': biome_details,
        'character_details': character_details,
        'plot_arcs': plot_arcs,
        'historical_evolution': historical_evolution
    }
    if use_tts and story:
        audio_path = text_to_speech(story)
        extra['audio'] = audio_path
    if use_sentiment and story:
        sentiment = analyze_sentiment(story)
        extra['sentiment'] = sentiment
    if db and story:
        story_id = f"story_{int(time.time())}_final"
        db.add_story(story_id, story, {"score": critique['overall_score'], "dna": world_dna[:50], **extra})
    return story, critique, extra  # Return best even if not perfect

if __name__ == "__main__":
    dna = load_world_dna("world_dna.md")
    idea, critique, _ = generate_story_idea(dna)
    print("Generated Story Idea:")
    print(idea)
    print("\nCritique:")
    print(critique)