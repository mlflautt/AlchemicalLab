"""
Graph Persistence System for AlchemicalLab.

Provides checkpointing, branching, and delta storage for:
- Knowledge Graph states
- CA terrain grids
- Audio presets
- World metadata
"""

import json
import os
import gzip
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import shutil


@dataclass
class WorldMetadata:
    """World metadata."""
    name: str
    version: str = "1.0"
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    modified: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    generation: int = 0
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class Checkpoint:
    """A checkpoint of world state."""
    checkpoint_id: str
    generation: int
    created: str
    file_path: str
    size_bytes: int
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Branch:
    """A timeline branch."""
    branch_id: str
    name: str
    created: str
    parent_checkpoint_id: str
    current_generation: int


class GraphPersistence:
    """
    Manages persistent storage of AlchemicalLab world states.
    
    Features:
    - Full state snapshots
    - Incremental checkpoints
    - Timeline branching
    - Delta compression
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.world_name = None
        self.world_path = None
        
        # Index of all checkpoints
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.branches: Dict[str, Branch] = {}
        self.current_branch = "main"
    
    def create_world(self, name: str, metadata: WorldMetadata = None) -> str:
        """Create a new world directory."""
        self.world_name = name
        self.world_path = self.base_path / name.replace(" ", "_")
        
        # Create directory structure
        self.world_path.mkdir(parents=True, exist_ok=True)
        (self.world_path / "checkpoints").mkdir(exist_ok=True)
        (self.world_path / "branches").mkdir(exist_ok=True)
        (self.world_path / "ca_states").mkdir(exist_ok=True)
        (self.world_path / "audio_cache").mkdir(exist_ok=True)
        
        # Write metadata
        if metadata is None:
            metadata = WorldMetadata(name=name)
        else:
            metadata.name = name
        
        self._write_metadata(metadata)
        
        return str(self.world_path)
    
    def _write_metadata(self, metadata: WorldMetadata):
        """Write world metadata."""
        meta_path = self.world_path / "meta.json"
        with open(meta_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
    
    def load_metadata(self, world_name: str) -> WorldMetadata:
        """Load world metadata."""
        world_path = self.base_path / world_name.replace(" ", "_")
        meta_path = world_path / "meta.json"
        
        with open(meta_path, 'r') as f:
            data = json.load(f)
        
        return WorldMetadata(**data)
    
    def save_checkpoint(
        self,
        graph_data: Dict,
        ca_grid: np.ndarray = None,
        audio_preset: Dict = None,
        generation: int = 0,
        metadata: Dict = None
    ) -> str:
        """Save a checkpoint of current state."""
        if self.world_path is None:
            raise RuntimeError("No world loaded. Call create_world() first.")
        
        # Generate checkpoint ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"gen{generation:06d}_{timestamp}"
        
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "generation": generation,
            "created": datetime.utcnow().isoformat(),
            "graph": graph_data,
            "audio_preset": audio_preset,
            "metadata": metadata or {},
        }
        
        # Compress and save
        checkpoint_file = self.world_path / "checkpoints" / f"{checkpoint_id}.json.gz"
        
        with gzip.open(checkpoint_file, 'wt') as f:
            json.dump(checkpoint_data, f)
        
        # Save CA grid as binary if provided
        if ca_grid is not None:
            ca_file = self.world_path / "ca_states" / f"{checkpoint_id}.npy"
            np.save(ca_file, ca_grid)
        
        # Create checkpoint index
        size_bytes = checkpoint_file.stat().st_size
        parent_id = self._get_current_checkpoint_id()
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            generation=generation,
            created=datetime.utcnow().isoformat(),
            file_path=str(checkpoint_file),
            size_bytes=size_bytes,
            parent_id=parent_id,
            metadata=metadata or {}
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        
        # Update current checkpoint reference
        self._write_current_checkpoint(checkpoint_id)
        
        return checkpoint_id
    
    def _get_current_checkpoint_id(self) -> Optional[str]:
        """Get the ID of the most recent checkpoint."""
        current_file = self.world_path / "checkpoints" / "_current"
        if current_file.exists():
            with open(current_file, 'r') as f:
                return f.read().strip()
        return None
    
    def _write_current_checkpoint(self, checkpoint_id: str):
        """Update current checkpoint reference."""
        current_file = self.world_path / "checkpoints" / "_current"
        with open(current_file, 'w') as f:
            f.write(checkpoint_id)
    
    def load_checkpoint(self, checkpoint_id: str) -> Dict:
        """Load a checkpoint by ID."""
        checkpoint_file = self.world_path / "checkpoints" / f"{checkpoint_id}.json.gz"
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        
        with gzip.open(checkpoint_file, 'rt') as f:
            data = json.load(f)
        
        return data
    
    def load_ca_state(self, checkpoint_id: str) -> Optional[np.ndarray]:
        """Load CA state for a checkpoint."""
        ca_file = self.world_path / "ca_states" / f"{checkpoint_id}.npy"
        
        if ca_file.exists():
            return np.load(ca_file)
        return None
    
    def create_branch(
        self,
        branch_name: str,
        from_checkpoint_id: str = None
    ) -> str:
        """Create a new timeline branch."""
        if self.world_path is None:
            raise RuntimeError("No world loaded.")
        
        if from_checkpoint_id is None:
            from_checkpoint_id = self._get_current_checkpoint_id()
            if from_checkpoint_id is None:
                raise ValueError("No checkpoint to branch from")
        
        # Generate branch ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        branch_id = f"{branch_name.replace(' ', '_').lower()}_{timestamp}"
        
        branch = Branch(
            branch_id=branch_id,
            name=branch_name,
            created=datetime.utcnow().isoformat(),
            parent_checkpoint_id=from_checkpoint_id,
            current_generation=self.checkpoints.get(from_checkpoint_id, Checkpoint("", 0, "", "", 0)).generation
        )
        
        # Save branch info
        branch_file = self.world_path / "branches" / f"{branch_id}.json"
        with open(branch_file, 'w') as f:
            json.dump(asdict(branch), f, indent=2)
        
        self.branches[branch_id] = branch
        
        return branch_id
    
    def switch_branch(self, branch_id: str):
        """Switch to a different branch."""
        branch_file = self.world_path / "branches" / f"{branch_id}.json"
        
        if not branch_file.exists():
            raise FileNotFoundError(f"Branch not found: {branch_id}")
        
        with open(branch_file, 'r') as f:
            branch_data = json.load(f)
        
        self.current_branch = branch_id
        self.branches[branch_id] = Branch(**branch_data)
    
    def list_branches(self) -> List[Branch]:
        """List all branches."""
        branch_files = (self.world_path / "branches").glob("*.json")
        
        branches = []
        for bf in branch_files:
            with open(bf, 'r') as f:
                branches.append(Branch(**json.load(f)))
        
        return sorted(branches, key=lambda b: b.created, reverse=True)
    
    def get_checkpoint_timeline(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent checkpoint timeline."""
        recent = sorted(
            self.checkpoints.values(),
            key=lambda c: c.created,
            reverse=True
        )[:limit]
        
        return [
            {
                "id": c.checkpoint_id,
                "generation": c.generation,
                "created": c.created,
                "size_kb": c.size_bytes / 1024,
            }
            for c in recent
        ]
    
    def save_delta(
        self,
        graph_delta: Dict,
        generation: int,
        base_checkpoint_id: str
    ) -> str:
        """Save only the changes since last checkpoint (delta)."""
        if self.world_path is None:
            raise RuntimeError("No world loaded.")
        
        delta_id = f"delta_gen{generation:06d}_{base_checkpoint_id}"
        
        delta_data = {
            "delta_id": delta_id,
            "base_checkpoint": base_checkpoint_id,
            "generation": generation,
            "created": datetime.utcnow().isoformat(),
            "changes": graph_delta,
        }
        
        delta_file = self.world_path / "checkpoints" / f"{delta_id}.json.gz"
        
        with gzip.open(delta_file, 'wt') as f:
            json.dump(delta_data, f)
        
        return delta_id
    
    def export_world(self, output_path: str, format: str = "zip"):
        """Export entire world to a file."""
        if format == "zip":
            shutil.make_archive(output_path.replace(".zip", ""), 'zip', self.world_path)
        elif format == "tar":
            shutil.make_archive(output_path.replace(".tar", ""), 'tar', self.world_path)
    
    def import_world(self, archive_path: str):
        """Import world from archive."""
        # Extract to temp location first
        temp_path = self.base_path / "_temp_import"
        shutil.unpack_archive(archive_path, temp_path)
        
        # Move to proper location
        world_dirs = list(temp_path.glob("*"))
        if world_dirs:
            world_name = world_dirs[0].name
            dest = self.base_path / world_name
            
            if dest.exists():
                shutil.rmtree(dest)
            
            shutil.move(str(world_dirs[0]), str(dest))
            
            # Cleanup
            shutil.rmtree(temp_path)
            
            self.world_name = world_name
            self.world_path = dest
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if self.world_path is None:
            return {}
        
        total_size = sum(
            f.stat().st_size 
            for f in self.world_path.rglob("*") 
            if f.is_file()
        )
        
        checkpoint_count = len(list((self.world_path / "checkpoints").glob("*.json.gz")))
        branch_count = len(list((self.world_path / "branches").glob("*.json")))
        
        ca_files = list((self.world_path / "ca_states").glob("*.npy"))
        ca_size = sum(f.stat().st_size for f in ca_files)
        
        return {
            "total_size_mb": total_size / (1024 * 1024),
            "checkpoint_count": checkpoint_count,
            "branch_count": branch_count,
            "ca_states_size_mb": ca_size / (1024 * 1024),
            "ca_state_count": len(ca_files),
        }


class DeltaEncoder:
    """Encodes and decodes graph deltas efficiently."""
    
    @staticmethod
    def encode_delta(old_data: Dict, new_data: Dict) -> Dict:
        """Compute delta between two states."""
        delta = {}
        
        # Check for new keys
        for key in new_data:
            if key not in old_data:
                delta[key] = {"op": "add", "value": new_data[key]}
            elif old_data[key] != new_data[key]:
                delta[key] = {"op": "update", "value": new_data[key]}
        
        # Check for deleted keys
        for key in old_data:
            if key not in new_data:
                delta[key] = {"op": "delete"}
        
        return delta
    
    @staticmethod
    def apply_delta(base_data: Dict, delta: Dict) -> Dict:
        """Apply delta to base data."""
        result = dict(base_data)
        
        for key, change in delta.items():
            if change["op"] == "add" or change["op"] == "update":
                result[key] = change["value"]
            elif change["op"] == "delete" and key in result:
                del result[key]
        
        return result


if __name__ == '__main__':
    # Test persistence
    print("Testing Graph Persistence...")
    
    persistence = GraphPersistence("./test_worlds")
    
    # Create world
    world_path = persistence.create_world("Test World")
    print(f"Created: {world_path}")
    
    # Save some checkpoints
    for gen in [0, 100, 200, 300]:
        checkpoint_id = persistence.save_checkpoint(
            graph_data={"nodes": gen, "edges": gen * 2},
            ca_grid=np.random.rand(32, 32),
            audio_preset={"name": "test", "frequency": 220},
            generation=gen,
            metadata={"note": f"Generation {gen}"}
        )
        print(f"Checkpoint: {checkpoint_id}")
    
    # Create a branch
    branch_id = persistence.create_branch("Alternative Timeline")
    print(f"Branch: {branch_id}")
    
    # List branches
    branches = persistence.list_branches()
    print(f"Branches: {[b.name for b in branches]}")
    
    # Get stats
    stats = persistence.get_statistics()
    print(f"Stats: {stats}")
    
    print("\nPersistence system working!")
