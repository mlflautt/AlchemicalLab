"""
Obsidian Sync for GraphEngine.

Provides bidirectional sync with Obsidian vault (Markdown files).
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import yaml
import re

from GraphEngine.core.types import KnowledgeNode, RelationDef


class ObsidianSync:
    """Bidirectional sync between KnowledgeGraph and Obsidian vault."""
    
    def __init__(
        self,
        vault_path: str,
        graph: 'KnowledgeGraph' = None
    ):
        """
        Initialize Obsidian sync.
        
        Args:
            vault_path: Path to Obsidian vault directory
            graph: Reference to KnowledgeGraph instance
        """
        self.vault_path = Path(vault_path)
        self.graph = graph
        
        # Ensure vault structure exists
        self._ensure_vault_structure()
    
    def _ensure_vault_structure(self):
        """Create Obsidian vault directory structure."""
        dirs = [
            "entities/characters",
            "entities/species",
            "entities/locations",
            "entities/factions",
            "concepts/themes",
            "concepts/patterns",
            "events/timeline",
            "_templates"
        ]
        
        for dir_path in dirs:
            (self.vault_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    def _safe_filename(self, name: str) -> str:
        """Convert name to safe filename."""
        # Replace unsafe characters
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Replace spaces with underscores
        safe = safe.replace(' ', '_')
        # Remove consecutive underscores
        safe = re.sub(r'_+', '_', safe)
        # Remove leading/trailing underscores
        safe = safe.strip('_')
        return safe.lower() or "unnamed"
    
    def _build_frontmatter(self, node: KnowledgeNode) -> str:
        """Build YAML frontmatter for node."""
        frontmatter = {
            'id': node.id,
            'type': node.type,
            'created': node.created.strftime('%Y-%m-%d'),
            'modified': node.modified.strftime('%Y-%m-%d'),
            'source': node.source,
            'tags': node.tags
        }
        
        if node.embedding_id:
            frontmatter['embedding_id'] = node.embedding_id
        
        # Add relations
        if node.relations:
            frontmatter['relations'] = [
                {
                    'to': f"[[{self._get_wikilink_name(r.to, r.type)}]]",
                    'type': r.type,
                    'weight': r.weight,
                    'context': r.context
                }
                for r in node.relations
            ]
        
        # Add properties as top-level (excluding name which becomes title)
        for key, value in node.properties.items():
            if key not in ['name', 'title']:
                frontmatter[key] = value
        
        return yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
    
    def _get_wikilink_name(self, node_id: str, edge_type: str) -> str:
        """Get wikilink display name for a node ID."""
        if self.graph:
            node = self.graph.get_node(node_id)
            if node:
                return node.get_name()
        return node_id
    
    def _build_content(self, node: KnowledgeNode) -> str:
        """Build markdown content for node."""
        lines = []
        
        # Title
        name = node.properties.get('name', node.properties.get('title', node.id))
        lines.append(f"# {name}")
        lines.append("")
        
        # Type-specific content
        if node.type == 'character':
            lines.extend(self._build_character_content(node))
        elif node.type == 'species':
            lines.extend(self._build_species_content(node))
        elif node.type == 'location':
            lines.extend(self._build_location_content(node))
        elif node.type == 'faction':
            lines.extend(self._build_faction_content(node))
        elif node.type == 'event':
            lines.extend(self._build_event_content(node))
        else:
            lines.extend(self._build_generic_content(node))
        
        return "\n".join(lines)
    
    def _build_character_content(self, node: KnowledgeNode) -> List[str]:
        """Build content for character nodes."""
        lines = []
        props = node.properties
        
        role = props.get('role', 'Unknown')
        motivation = props.get('motivation', '')
        lines.append(f"**Role:** {role} • **Motivation:** {motivation}")
        lines.append("")
        
        if 'appearance' in props:
            lines.append("## Appearance")
            lines.append(props['appearance'])
            lines.append("")
        
        if 'backstory' in props:
            lines.append("## Backstory")
            lines.append(props['backstory'])
            lines.append("")
        
        if node.relations:
            lines.append("## Relationships")
            for rel in node.relations:
                target_name = self._get_wikilink_name(rel.to, rel.type)
                lines.append(f"- [[{target_name}]] - {rel.type}" + (f" ({rel.context})" if rel.context else ""))
            lines.append("")
        
        if 'secrets' in props:
            lines.append("## Secrets & Hooks")
            lines.append(props['secrets'])
        
        return lines
    
    def _build_species_content(self, node: KnowledgeNode) -> List[str]:
        """Build content for species nodes."""
        lines = []
        props = node.properties
        
        species_type = props.get('species_type', 'unknown')
        population = props.get('population', 0)
        fitness = props.get('fitness', 0)
        
        lines.append(f"**Type:** {species_type} • **Population:** {population} • **Fitness:** {fitness:.2f}")
        lines.append("")
        
        if 'traits' in props:
            lines.append("## Traits")
            for trait, value in props['traits'].items():
                lines.append(f"- {trait}: {value}")
            lines.append("")
        
        if 'preferred_biomes' in props:
            lines.append("## Preferred Biomes")
            lines.append(", ".join(props['preferred_biomes']))
            lines.append("")
        
        if node.relations:
            lines.append("## Ecological Relationships")
            for rel in node.relations:
                target_name = self._get_wikilink_name(rel.to, rel.type)
                lines.append(f"- [[{target_name}]] ({rel.type}): weight {rel.weight:.2f}")
        
        return lines
    
    def _build_location_content(self, node: KnowledgeNode) -> List[str]:
        """Build content for location nodes."""
        lines = []
        props = node.properties
        
        location_type = props.get('location_type', 'unknown')
        atmosphere = props.get('atmosphere', 'neutral')
        
        lines.append(f"**Type:** {location_type} • **Atmosphere:** {atmosphere}")
        lines.append("")
        
        if 'description' in props:
            lines.append("## Description")
            lines.append(props['description'])
            lines.append("")
        
        if 'resources' in props:
            lines.append("## Resources")
            lines.append(props['resources'])
        
        return lines
    
    def _build_faction_content(self, node: KnowledgeNode) -> List[str]:
        """Build content for faction nodes."""
        lines = []
        props = node.properties
        
        ideology = props.get('ideology', 'unknown')
        methods = props.get('methods', '')
        
        lines.append(f"**Ideology:** {ideology} • **Methods:** {methods}")
        lines.append("")
        
        if 'goals' in props:
            lines.append("## Goals")
            lines.append(props['goals'])
            lines.append("")
        
        if 'members' in props:
            lines.append("## Members")
            for member in props['members']:
                lines.append(f"- [[{member}]]")
            lines.append("")
        
        if 'territories' in props:
            lines.append("## Territories")
            for territory in props['territories']:
                lines.append(f"- [[{territory}]]")
        
        return lines
    
    def _build_event_content(self, node: KnowledgeNode) -> List[str]:
        """Build content for event nodes."""
        lines = []
        props = node.properties
        
        event_type = props.get('event_type', 'unknown')
        timestamp = props.get('timestamp', 'unknown')
        
        lines.append(f"**Type:** {event_type} • **Time:** {timestamp}")
        lines.append("")
        
        if 'description' in props:
            lines.append("## Description")
            lines.append(props['description'])
            lines.append("")
        
        if 'participants' in props:
            lines.append("## Participants")
            for participant in props['participants']:
                lines.append(f"- [[{participant}]]")
        
        return lines
    
    def _build_generic_content(self, node: KnowledgeNode) -> List[str]:
        """Build content for generic nodes."""
        lines = []
        
        for key, value in node.properties.items():
            if key not in ['name', 'title']:
                lines.append(f"**{key.title()}:** {value}")
        
        return lines
    
    def export_node(self, node_id: str) -> Optional[Path]:
        """
        Export a single node to Obsidian MD file.
        
        Returns:
            Path to created file, or None if failed
        """
        if not self.graph:
            return None
        
        node = self.graph.get_node(node_id)
        if not node:
            return None
        
        # Determine file path
        type_dir = self.vault_path / "entities" / f"{node.type}s"
        type_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self._safe_filename(node.get_name())}.md"
        file_path = type_dir / filename
        
        # Build file content
        frontmatter = self._build_frontmatter(node)
        content = self._build_content(node)
        
        # Write file
        file_path.write_text(f"---\n{frontmatter}---\n\n{content}")
        
        return file_path
    
    def export_all(self) -> List[str]:
        """Export all nodes to Obsidian vault."""
        if not self.graph:
            return []
        
        exported = []
        
        # Export all nodes
        node_ids = self.graph.list_node_ids()
        
        for node_id in node_ids:
            path = self.export_node(node_id)
            if path:
                exported.append(str(path))
        
        return exported
    
    def import_file(self, md_path: Path) -> Optional[str]:
        """
        Import a single Obsidian MD file to graph.
        
        Returns:
            Node ID if successful, None if failed
        """
        if not self.graph:
            return None
        
        try:
            content = md_path.read_text()
            
            # Parse frontmatter
            if not content.startswith('---'):
                return None
            
            # Split frontmatter and content
            parts = content.split('---', 2)
            if len(parts) < 3:
                return None
            
            frontmatter_text = parts[1].strip()
            body = parts[2].strip()
            
            # Parse YAML frontmatter
            frontmatter = yaml.safe_load(frontmatter_text)
            
            if not isinstance(frontmatter, dict):
                return None
            
            # Extract properties
            node_type = frontmatter.pop('type', 'concept')
            node_id = frontmatter.pop('id', None)
            tags = frontmatter.pop('tags', [])
            source = frontmatter.pop('source', 'user_created')
            
            # Parse relations
            relations_raw = frontmatter.pop('relations', [])
            relations = []
            
            for rel in relations_raw:
                if isinstance(rel, dict):
                    to = rel.get('to', '')
                    # Extract node ID from wikilink
                    if to.startswith('[[') and to.endswith(']]'):
                        to = to[2:-2]
                    
                    relations.append(RelationDef(
                        to=to,
                        type=rel.get('type', 'references'),
                        weight=rel.get('weight', 1.0),
                        context=rel.get('context', '')
                    ))
            
            # Extract name from body or frontmatter
            name = frontmatter.pop('name', None)
            if not name:
                # Try to extract from first heading
                for line in body.split('\n'):
                    if line.startswith('# '):
                        name = line[2:].strip()
                        break
            
            if not name:
                name = md_path.stem
            
            # Build properties
            properties = {'name': name}
            properties.update(frontmatter)
            
            # Create node
            node_id = self.graph.add_node(
                node_type=node_type,
                properties=properties,
                relations=relations,
                tags=tags,
                source=source,
                node_id=node_id
            )
            
            return node_id
            
        except Exception as e:
            print(f"Error importing {md_path}: {e}")
            return None
    
    def import_vault(self) -> List[str]:
        """
        Import all MD files from vault to graph.
        
        Returns:
            List of imported node IDs
        """
        imported = []
        
        # Find all MD files
        for md_file in self.vault_path.rglob("*.md"):
            # Skip templates and hidden files
            if md_file.name.startswith('_') or md_file.name.startswith('.'):
                continue
            
            node_id = self.import_file(md_file)
            if node_id:
                imported.append(node_id)
        
        return imported
    
    def sync(self) -> Dict[str, Any]:
        """
        Perform bidirectional sync.
        
        Returns:
            Dict with sync statistics
        """
        stats = {
            'exported': 0,
            'imported': 0,
            'errors': []
        }
        
        # Export new/modified nodes
        if self.graph:
            exported = self.export_all()
            stats['exported'] = len(exported)
        
        # Import new files
        imported = self.import_vault()
        stats['imported'] = len(imported)
        
        return stats
