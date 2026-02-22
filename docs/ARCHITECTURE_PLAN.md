# Generative Story/World-Building Architecture Plan

**Author**: opencode/glm-5-free  
**Date**: 2026-02-22  
**Status**: Active Development

---

## Telos

A knowledge-graph-centered architecture where multiple generative systems interact deterministically to produce consistent, emergent narratives.

## Core Philosophy

1. **Graph as Single Source of Truth**: All entities, relationships, and events live in the knowledge graph
2. **Deterministic Context Extraction**: Same entity + task = same context (reproducible, debuggable)
3. **Token Budget Awareness**: Context extraction respects LLM context window limits
4. **CA as Graph Evolution**: Cellular automata rules operate on graph nodes/edges
5. **Consistency First**: All generated content validated before graph insertion
6. **Fractal Initialization**: Graph topology can be fractally structured for emergent variety

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        KNOWLEDGE GRAPH (Core)                            │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Context Extraction Layer                       │   │
│   │   get_context(entity, task_type, max_tokens) → prompt_ready      │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│   │ species  │ │character │ │ faction  │ │  event   │ │ concept  │ ... │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘     │
│        └────────────┴────────────┴────────────┴────────────┘            │
│                           edges (relationships)                          │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│    CALab      │          │   StoryLab    │          │ Future Gen    │
│ (Emergence)   │          │ (LLM Content) │          │ Systems       │
├───────────────┤          ├───────────────┤          ├───────────────┤
│ • GraphCA     │          │ • Character   │          │ • Character   │
│   (evolve     │          │   Generation  │          │   Agents      │
│   nodes/edges)│          │ • Scene       │          │ • World Sim   │
│ • Species     │          │   Generation  │          │ • Consistency │
│   Evolution   │          │ • Dialogue    │          │   Checker     │
│ • Pattern     │          │   Generation  │          │ • Narrative   │
│   Detection   │          │ • Critic Loop │          │   Planner     │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
        └──────────────────────────┴──────────────────────────┘
                                   │
                           WRITE TO GRAPH
                        (consistent storage)
```

---

## Phase 2: Context Extraction Layer

### Goal
Deterministic subgraph extraction for LLM prompts with token budget awareness.

### Components

| Component | Purpose | File |
|-----------|---------|------|
| `ContextExtractor` | Budget-aware context extraction | `GraphEngine/context/extractor.py` |
| `ContextTemplates` | Task-specific context schemas | `GraphEngine/context/templates.py` |
| `PromptSerializer` | Graph → prompt format conversion | `GraphEngine/context/serializer.py` |

### Task Types & Context Requirements

| Task Type | Focus | Context Includes | Max Nodes | Typical Tokens |
|-----------|-------|------------------|-----------|----------------|
| `scene_generation` | Location + characters | Present characters, location, recent events, active conflicts | 15 | 1500-2500 |
| `dialogue` | Speaker + listener | Both characters, relationship, shared history, current situation | 5 | 500-1000 |
| `character_decision` | Single character | Character's knowledge, goals, constraints, available actions | 10 | 800-1500 |
| `faction_conflict` | Two factions | Both factions, key members, territories, history, resources | 20 | 2000-3000 |
| `world_event` | Event trigger | Affected entities, causal chain, location, timing | 25 | 1500-2500 |
| `character_creation` | World context | Existing factions, locations, species, themes, conflicts | 30 | 2000-3500 |
| `location_description` | Location | Location properties, inhabitants, recent events, connections | 10 | 500-1000 |

### API Design

```python
from GraphEngine.context import ContextExtractor

extractor = ContextExtractor(graph)

# Basic context extraction
context = extractor.get_context(
    entity_id="char_elena",
    task_type="scene_generation",
    max_tokens=2000
)

# Returns:
# {
#   'focus_entity': KnowledgeNode,
#   'related_entities': [KnowledgeNode, ...],
#   'relationships': [KnowledgeEdge, ...],
#   'recent_events': [KnowledgeNode, ...],
#   'active_conflicts': [Conflict, ...],
#   'temporal_context': {...},
#   'token_count': int,
#   'extraction_metadata': {...}
# }

# Serialize for LLM prompt
prompt_context = extractor.serialize(context, format='markdown')
```

### Context Extraction Algorithm

```
1. Start with focus entity (the entity we're generating content for)
2. Add immediate relationships (depth=1 neighbors)
3. Add temporal context (recent events involving focus)
4. Add conflict context (active tensions affecting focus)
5. Prune by importance (keep high-importance nodes)
6. Truncate to token budget (while maintaining coherence)
7. Return structured context with metadata
```

### Importance Heuristics

| Factor | Weight | Rationale |
|--------|--------|-----------|
| Edge strength | 0.3 | Stronger relationships = more relevant |
| Node recency | 0.2 | Recent nodes more relevant |
| Node importance property | 0.2 | Explicit importance marker |
| Relationship type | 0.15 | Conflict/alliance > references |
| Distance from focus | 0.15 | Closer = more relevant |

---

## Phase 3: Graph-Based CA Evolution

### Goal
CA operates directly on graph nodes/edges for emergent evolution of world state.

### Concept: Graph CA

```
Traditional CA:  cells[i,j] → update based on 8 neighbors
Graph CA:        nodes[id] → update based on connected nodes via edges

Cell state → Node properties
Neighborhood → Connected nodes via edge types
Update rules → Node/edge transformation functions
```

### Components

| Component | Purpose | File |
|-----------|---------|------|
| `GraphCA` | CA engine for graph evolution | `CALab/graph_ca/core.py` |
| `EvolutionRules` | Node/edge update rules | `CALab/graph_ca/rules.py` |
| `EmergenceDetector` | Pattern detection in evolving graph | `CALab/graph_ca/emergence.py` |

### Evolution Rules

#### Species Evolution Rules
```python
# Population dynamics
population_next = population_current * fitness
- predation_weight * sum(prey.population for prey in predators)
+ consumption_weight * sum(prey.population for prey in food_sources)

# Fitness adaptation
fitness_next = fitness_current + mutation_rate * random()
- competition_penalty * competitors_in_territory
```

#### Faction Evolution Rules
```python
# Power dynamics
power_next = power_current
+ territory_count * territory_value
+ member_count * member_strength
- conflicts_lost * conflict_cost

# Territory expansion (if power > threshold)
new_territory = adjacent_unclaimed_territory
```

#### Character Evolution Rules
```python
# Stress accumulation
stress_next = stress_current
+ conflict_weight * active_conflicts
- ally_weight * ally_count
- event_weight * positive_events

# Belief updates (based on observed events)
beliefs.update(observed_event)
```

### Emergence Detection

| Pattern | Detection | Result |
|---------|-----------|--------|
| Population boom | population > 2x average | Create "expansion" event |
| Extinction | population < threshold | Create "extinction" event, cascade effects |
| War eruption | conflict_weight > threshold | Create "war" event |
| Alliance formation | mutual_benefit > threshold | Create "alliance" edge |
| Power shift | faction_power change > 0.3 | Create "power_shift" event |

---

## Phase 4: Additional Generative Systems

### 4a. Character Agents

**Purpose**: Simulated character reasoning, memory, and decision-making.

**Components**:
- `CharacterAgent`: Per-character simulation instance
- `MemoryStore`: Limited-perspective knowledge store
- `DecisionEngine`: Goal-directed action selection

**Key Methods**:
```python
agent = CharacterAgent(graph, "char_elena")

# What does this character know?
knowledge = agent.get_knowledge()  # Limited perspective subgraph

# Make a decision in a situation
decision = agent.make_decision(situation)

# Learn about an event
agent.observe(event)
```

### 4b. World Simulator

**Purpose**: Generate events from world state tensions.

**Components**:
- `TensionDetector`: Find conflicts, resource shortages, imbalances
- `EventGenerator`: Create events addressing tensions
- `CascadeResolver`: Propagate event effects

**Key Methods**:
```python
simulator = WorldSimulator(graph)

# Find tensions in the world
tensions = simulator.detect_tensions()

# Generate event from tension
event = simulator.generate_event(tensions[0])

# Resolve cascade effects
effects = simulator.resolve_cascade(event)
```

### 4c. Consistency Checker

**Purpose**: Validate generated content against graph constraints.

**Constraint Types**:
| Constraint | Description | Validation |
|------------|-------------|------------|
| `character_location` | Character at exactly one location | Check location edges |
| `temporal_causality` | Causes precede effects | Check timestamps |
| `relationship_symmetry` | Bidirectional edges symmetric | Check both directions |
| `power_balance` | Power changes have causes | Trace causal chain |
| `knowledge_scope` | Characters know what they've observed | Check event exposure |

**Key Methods**:
```python
checker = ConsistencyChecker(graph)

# Validate content
result = checker.validate(generated_content)

if not result.is_valid:
    fixes = checker.suggest_fixes(result.violations)
```

### 4d. Narrative Planner

**Purpose**: Long-term plot structure and foreshadowing.

**Components**:
- `ArcGenerator`: Multi-episode plot arc creation
- `ForeshadowingTracker`: Early hints for future events
- `TensionCurveManager`: Monitor and adjust narrative tension

---

## Phase 5: Enhanced StoryLab Integration

### Current Issues
1. world_dna truncated to 500 chars → massive context loss
2. LLM outputs not stored in graph
3. No feedback loop from generation to CA evolution

### New Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Generation Pipeline                       │
│                                                              │
│  1. EXTRACT context from graph (deterministic)              │
│     ↓                                                        │
│  2. GENERATE content via LLM with context                    │
│     ↓                                                        │
│  3. PARSE generated content into structured data             │
│     ↓                                                        │
│  4. VALIDATE against graph constraints                       │
│     ↓ (if valid)                                             │
│  5. STORE new nodes/edges in graph                           │
│     ↓                                                        │
│  6. EVOLVE graph via CA rules (triggered by changes)         │
│     ↓                                                        │
│  7. DETECT emergence (new events, conflicts, etc.)           │
│     ↓                                                        │
│  ┌─→ Return to step 1 for next generation                    │
│  │                                                           │
│  └───────────────────────────────────────────────────────────┘
```

### Implementation Updates

1. **Replace `world_dna` truncation** with `ContextExtractor.get_context()`
2. **Store LLM outputs** via `StoryLabBridge.process_generated_content()`
3. **Trigger CA evolution** after content insertion
4. **Validate before insertion** with `ConsistencyChecker`

---

## Implementation Order

| Phase | Component | Priority | Effort | Dependencies |
|-------|-----------|----------|--------|--------------|
| 2 | ContextExtractor | HIGH | 2 days | None |
| 2 | ContextTemplates | HIGH | 1 day | ContextExtractor |
| 2 | PromptSerializer | HIGH | 1 day | ContextExtractor |
| 4c | ConsistencyChecker | HIGH | 1 day | ContextExtractor |
| 3 | GraphCA Core | HIGH | 2 days | ContextExtractor |
| 3 | EvolutionRules | HIGH | 1 day | GraphCA Core |
| 3 | EmergenceDetector | HIGH | 1 day | GraphCA Core |
| 5 | Enhanced StoryLab | MEDIUM | 2 days | All above |
| 4a | Character Agents | MEDIUM | 2 days | ContextExtractor |
| 4b | World Simulator | MEDIUM | 2 days | GraphCA |
| 4d | Narrative Planner | LOW | 2 days | All above |

---

## Data Flow Summary

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW                                        │
│                                                                           │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  CA     │───▶│  GraphCA    │───▶│ Knowledge   │───▶│  Context    │   │
│  │ Rules   │    │  Evolution  │    │   Graph     │    │ Extractor   │   │
│  └─────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘   │
│                                           ▲                   │          │
│                                           │                   ▼          │
│  ┌─────────┐    ┌─────────────┐    ┌──────┴──────┐    ┌─────────────┐   │
│  │  LLM    │◀───│   Prompt    │◀───│  Serialize  │◀───│   Context   │   │
│  │ Model   │    │  Generator  │    │   to Text   │    │   Package   │   │
│  └────┬────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐                       │
│  │ Output  │───▶│ Consistency │───▶│  Store in   │───▶ Graph             │
│  │ Parser  │    │  Checker    │    │   Graph     │                       │
│  └─────────┘    └─────────────┘    └─────────────┘                       │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

1. **Context Coherence**: Generated content references correct entities/relationships
2. **Constraint Compliance**: < 5% of generated content rejected by consistency checker
3. **Emergence Quality**: CA evolution produces meaningful events (not noise)
4. **Token Efficiency**: Context extraction uses < 80% of token budget typically
5. **Reproducibility**: Same input → same output (deterministic)

---

## Future Extensions

- **Multi-model Ensemble**: Different LLMs for different task types
- **Interactive Mode**: Real-time user intervention in generation
- **Export Formats**: Novel format, screenplay, game script
- **Visualization**: Real-time graph evolution viewer
- **Learning**: Rules improve from feedback
