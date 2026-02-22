"""
Prompt Serializer for GraphEngine.

Converts extracted context into LLM-ready prompt formats.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class PromptSerializer:
    """Serialize graph context to various prompt formats."""
    
    @staticmethod
    def serialize(
        context: Dict[str, Any],
        format: str = 'markdown',
        task_type: str = None
    ) -> str:
        """
        Serialize context to specified format.
        
        Args:
            context: Extracted context dict
            format: Output format ('markdown', 'json', 'narrative', 'compact')
            task_type: Optional task type for format customization
        
        Returns:
            Formatted string ready for LLM prompt
        """
        serializers = {
            'markdown': PromptSerializer._to_markdown,
            'json': PromptSerializer._to_json,
            'narrative': PromptSerializer._to_narrative,
            'compact': PromptSerializer._to_compact,
        }
        
        serializer = serializers.get(format, PromptSerializer._to_markdown)
        return serializer(context, task_type)
    
    @staticmethod
    def _to_markdown(context: Dict[str, Any], task_type: str = None) -> str:
        """Convert context to markdown format."""
        sections = []
        
        sections.append("# Context for Generation\n")
        
        if task_type:
            sections.append(f"**Task:** {task_type}\n")
        
        focus = context.get('focus_entity')
        if focus:
            sections.append("## Focus Entity\n")
            sections.append(PromptSerializer._format_entity(focus))
            sections.append("")
        
        related = context.get('related_entities', [])
        if related:
            sections.append("## Related Entities\n")
            for entity in related[:10]:
                sections.append(PromptSerializer._format_entity_brief(entity))
            if len(related) > 10:
                sections.append(f"... and {len(related) - 10} more\n")
            sections.append("")
        
        relationships = context.get('relationships', [])
        if relationships:
            sections.append("## Relationships\n")
            for rel in relationships[:15]:
                sections.append(PromptSerializer._format_relationship(rel))
            sections.append("")
        
        events = context.get('recent_events', [])
        if events:
            sections.append("## Recent Events\n")
            for event in events[:10]:
                sections.append(PromptSerializer._format_event(event))
            sections.append("")
        
        conflicts = context.get('active_conflicts', [])
        if conflicts:
            sections.append("## Active Conflicts\n")
            for conflict in conflicts[:5]:
                sections.append(PromptSerializer._format_conflict(conflict))
            sections.append("")
        
        world_state = context.get('world_state', {})
        if world_state:
            sections.append("## World State Summary\n")
            sections.append(PromptSerializer._format_world_state(world_state))
        
        metadata = context.get('extraction_metadata', {})
        if metadata:
            sections.append("\n---\n")
            sections.append(f"*Context extracted: {metadata.get('timestamp', 'unknown')}*")
            sections.append(f" *Token estimate: {context.get('token_count', 'unknown')}*")
        
        return "\n".join(sections)
    
    @staticmethod
    def _to_json(context: Dict[str, Any], task_type: str = None) -> str:
        """Convert context to JSON format."""
        output = {
            'task_type': task_type,
            'focus_entity': PromptSerializer._entity_to_dict(context.get('focus_entity')),
            'related_entities': [
                PromptSerializer._entity_to_dict(e) 
                for e in context.get('related_entities', [])
            ],
            'relationships': [
                PromptSerializer._edge_to_dict(e)
                for e in context.get('relationships', [])
            ],
            'recent_events': [
                PromptSerializer._entity_to_dict(e)
                for e in context.get('recent_events', [])
            ],
            'active_conflicts': context.get('active_conflicts', []),
            'world_state': context.get('world_state', {}),
            'metadata': {
                'token_count': context.get('token_count'),
                'extraction_metadata': context.get('extraction_metadata', {})
            }
        }
        
        return json.dumps(output, indent=2, default=str)
    
    @staticmethod
    def _to_narrative(context: Dict[str, Any], task_type: str = None) -> str:
        """Convert context to narrative prose format."""
        lines = []
        
        focus = context.get('focus_entity')
        if focus:
            name = focus.properties.get('name', focus.id)
            entity_type = focus.type
            lines.append(f"This is the story context for {name}, a {entity_type}.")
            lines.append("")
            
            if entity_type == 'character':
                role = focus.properties.get('role', 'unknown role')
                motivation = focus.properties.get('motivation', 'unknown motivation')
                lines.append(f"{name} is a {role}, driven by {motivation}.")
                
                backstory = focus.properties.get('backstory')
                if backstory:
                    lines.append(f"Backstory: {backstory}")
            
            elif entity_type == 'faction':
                ideology = focus.properties.get('ideology', 'unknown ideology')
                goals = focus.properties.get('goals', 'unknown goals')
                lines.append(f"{name} follows the ideology of {ideology}, with goals to {goals}.")
        
        related = context.get('related_entities', [])
        if related:
            lines.append("")
            lines.append("Key figures and places in this context:")
            for entity in related[:5]:
                name = entity.properties.get('name', entity.id)
                rel_type = entity.type
                lines.append(f"- {name} ({rel_type})")
        
        relationships = context.get('relationships', [])
        if relationships:
            lines.append("")
            lines.append("Important relationships:")
            for rel in relationships[:5]:
                source = rel.get('source_name', rel.get('source_id', '?'))
                target = rel.get('target_name', rel.get('target_id', '?'))
                rel_type = rel.get('edge_type', 'connected to')
                lines.append(f"- {source} {rel_type} {target}")
        
        events = context.get('recent_events', [])
        if events:
            lines.append("")
            lines.append("Recent history:")
            for event in events[:3]:
                name = event.properties.get('name', event.id)
                description = event.properties.get('description', '')
                lines.append(f"- {name}: {description}" if description else f"- {name}")
        
        conflicts = context.get('active_conflicts', [])
        if conflicts:
            lines.append("")
            lines.append("Current tensions:")
            for conflict in conflicts[:3]:
                lines.append(f"- {conflict.get('description', 'Ongoing conflict')}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _to_compact(context: Dict[str, Any], task_type: str = None) -> str:
        """Convert context to compact one-line-per-entity format."""
        lines = []
        
        focus = context.get('focus_entity')
        if focus:
            lines.append(f"FOCUS: {focus.type}:{focus.properties.get('name', focus.id)}")
        
        related = context.get('related_entities', [])
        for entity in related[:15]:
            name = entity.properties.get('name', entity.id)
            etype = entity.type
            importance = entity.properties.get('importance', 0)
            lines.append(f"ENTITY: {etype}:{name} (imp:{importance:.1f})")
        
        relationships = context.get('relationships', [])
        for rel in relationships[:15]:
            source = rel.get('source_name', rel.get('source_id', '?'))
            target = rel.get('target_name', rel.get('target_id', '?'))
            rel_type = rel.get('edge_type', 'rel')
            weight = rel.get('weight', 1.0)
            lines.append(f"REL: {source} --[{rel_type}:{weight:.1f}]--> {target}")
        
        events = context.get('recent_events', [])
        for event in events[:5]:
            name = event.properties.get('name', event.id)
            etype = event.properties.get('event_type', 'event')
            lines.append(f"EVENT: {etype}:{name}")
        
        conflicts = context.get('active_conflicts', [])
        for conflict in conflicts[:3]:
            lines.append(f"CONFLICT: {conflict.get('description', 'active')}")
        
        lines.append(f"TOKENS: ~{context.get('token_count', '?')}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_entity(entity) -> str:
        """Format a single entity for markdown."""
        if not entity:
            return ""
        
        name = entity.properties.get('name', entity.id)
        lines = [f"**{name}** ({entity.type})"]
        
        for key, value in entity.properties.items():
            if key not in ['name', 'title'] and value is not None:
                if isinstance(value, (list, dict)):
                    value_str = json.dumps(value, default=str)[:100]
                    if len(json.dumps(value, default=str)) > 100:
                        value_str += "..."
                else:
                    value_str = str(value)
                lines.append(f"- {key}: {value_str}")
        
        if entity.tags:
            lines.append(f"- tags: {', '.join(entity.tags)}")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_entity_brief(entity) -> str:
        """Format entity briefly."""
        if not entity:
            return ""
        
        name = entity.properties.get('name', entity.id)
        etype = entity.type
        role = entity.properties.get('role', entity.properties.get('ideology', ''))
        
        if role:
            return f"- **{name}** ({etype}): {role}\n"
        return f"- **{name}** ({etype})\n"
    
    @staticmethod
    def _format_relationship(rel: Dict) -> str:
        """Format a relationship."""
        source = rel.get('source_name', rel.get('source_id', '?'))
        target = rel.get('target_name', rel.get('target_id', '?'))
        rel_type = rel.get('edge_type', 'connected')
        weight = rel.get('weight')
        context = rel.get('context', '')
        
        line = f"- {source} --[{rel_type}]--> {target}"
        if weight is not None:
            line += f" (strength: {weight:.1f})"
        if context:
            line += f" | {context}"
        
        return line + "\n"
    
    @staticmethod
    def _format_event(event) -> str:
        """Format an event."""
        if not event:
            return ""
        
        name = event.properties.get('name', event.id)
        etype = event.properties.get('event_type', 'event')
        timestamp = event.properties.get('timestamp', event.properties.get('generation', ''))
        description = event.properties.get('description', '')
        
        line = f"- **{name}** ({etype})"
        if timestamp:
            line += f" [{timestamp}]"
        if description:
            line += f": {description[:100]}"
            if len(description) > 100:
                line += "..."
        
        return line + "\n"
    
    @staticmethod
    def _format_conflict(conflict: Dict) -> str:
        """Format a conflict."""
        parties = conflict.get('parties', [])
        description = conflict.get('description', 'Ongoing conflict')
        intensity = conflict.get('intensity', 'unknown')
        
        line = f"- {description}"
        if parties:
            line += f" (involving: {', '.join(parties[:3])})"
        if intensity:
            line += f" [{intensity}]"
        
        return line + "\n"
    
    @staticmethod
    def _format_world_state(world_state: Dict) -> str:
        """Format world state summary."""
        lines = []
        
        if 'total_characters' in world_state:
            lines.append(f"- Characters: {world_state['total_characters']}")
        if 'total_factions' in world_state:
            lines.append(f"- Factions: {world_state['total_factions']}")
        if 'total_locations' in world_state:
            lines.append(f"- Locations: {world_state['total_locations']}")
        if 'active_conflicts' in world_state:
            lines.append(f"- Active Conflicts: {world_state['active_conflicts']}")
        if 'generation' in world_state:
            lines.append(f"- World Generation: {world_state['generation']}")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _entity_to_dict(entity) -> Dict:
        """Convert entity to dictionary."""
        if not entity:
            return None
        
        return {
            'id': entity.id,
            'type': entity.type,
            'name': entity.properties.get('name', entity.id),
            'properties': {
                k: v for k, v in entity.properties.items()
                if k != 'name'
            },
            'tags': entity.tags,
            'source': entity.source,
        }
    
    @staticmethod
    def _edge_to_dict(edge) -> Dict:
        """Convert edge to dictionary."""
        if isinstance(edge, dict):
            return edge
        return {
            'source_id': edge.source_id,
            'target_id': edge.target_id,
            'edge_type': edge.edge_type,
            'weight': edge.weight,
            'context': edge.context,
        }


def estimate_token_count(text: str) -> int:
    """Estimate token count for text (rough approximation)."""
    return len(text) // 4 + len(text.split()) // 2


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token budget."""
    estimated = estimate_token_count(text)
    
    if estimated <= max_tokens:
        return text
    
    ratio = max_tokens / estimated
    target_chars = int(len(text) * ratio * 0.9)
    
    truncated = text[:target_chars]
    
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n\n')
    cut_point = max(last_period, last_newline)
    
    if cut_point > target_chars * 0.7:
        truncated = truncated[:cut_point + 1]
    
    return truncated + "\n\n[Context truncated to fit token budget]"
