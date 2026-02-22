#!/usr/bin/env python3
"""
Critic System: Validates generated story ideas for coherence and quality.
Combines rule-based checks with LLM-based critique.
"""

import subprocess
import re

def rule_based_critic(story_output, world_dna):
    """
    Rule-based checks for story coherence.
    Returns a dict with issues and score (0-10).
    """
    issues = []
    score = 10  # Start high, deduct for issues

    # Check for required sections
    required_sections = ["Main plot arc", "Key characters", "Interconnected subplots", "Potential conflicts"]
    for section in required_sections:
        if section.lower() not in story_output.lower():
            issues.append(f"Missing section: {section}")
            score -= 2

    # Check protagonist consistency
    if "young engineer" in world_dna.lower():
        if "young engineer" not in story_output.lower() and "engineer" not in story_output.lower():
            issues.append("Protagonist archetype not aligned (missing young engineer)")
            score -= 1

    # Check for factions
    factions = ["Arcanists", "Wanderers", "Voidborn"]
    faction_count = sum(1 for f in factions if f.lower() in story_output.lower())
    if faction_count < 2:
        issues.append("Too few factions referenced")
        score -= 1

    # Check length (rough proxy for depth)
    if len(story_output.split()) < 100:
        issues.append("Story too short, lacks depth")
        score -= 2

    score = max(0, score)  # Ensure non-negative
    return {"issues": issues, "score": score}

def llm_critic(story_output, world_dna, model_path="../qwen_q2.gguf", ngl=22):
    """
    LLM-based critique using Qwen.
    Returns critique text.
    """
    prompt = f"""
Critique this generated story idea for coherence, depth, and alignment with the world-DNA. Rate on a scale of 1-10 and suggest improvements.

World-DNA:
{world_dna}

Generated Story:
{story_output}

Provide a concise critique with score and suggestions.
"""
    cmd = ['../llama.cpp/build/bin/llama-simple', '-m', model_path, '-ngl', str(ngl), '-n', '256', prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    return result.stdout.strip()

def combined_critic(story_output, world_dna):
    """
    Combines rule-based and LLM critic.
    Returns overall score and feedback.
    """
    rules = rule_based_critic(story_output, world_dna)
    llm_feedback = llm_critic(story_output, world_dna)

    # Extract score from LLM if possible
    llm_score_match = re.search(r'(\d+)/10|score[:\s]*(\d+)', llm_feedback, re.IGNORECASE)
    llm_score = int(llm_score_match.group(1) or llm_score_match.group(2)) if llm_score_match else 7

    overall_score = (rules["score"] + llm_score) / 2
    feedback = {
        "rule_issues": rules["issues"],
        "llm_feedback": llm_feedback,
        "overall_score": overall_score
    }
    return feedback

if __name__ == "__main__":
    # Test with sample
    sample_story = "Main plot: Hero saves world. Characters: Hero, Villain."
    sample_dna = "Young engineer discovers artifact."
    result = combined_critic(sample_story, sample_dna)
    print("Critic Result:", result)