"""
Base Processing Module for GraphEngine.

Inspired by BrainSimII's module architecture where each module is a 
functional processing unit that operates on the knowledge graph.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from GraphEngine.core.types import ModuleResult


class ProcessingModule(ABC):
    """
    Abstract base class for processing modules.
    
    Modules are functional units that process the knowledge graph,
    similar to BrainSimII's processing modules. Each module:
    - Can read and modify nodes/edges
    - Produces a ModuleResult describing changes
    - Can be chained in a processing pipeline
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    def process(self, graph: 'KnowledgeGraph', **kwargs) -> ModuleResult:
        """
        Process the knowledge graph.
        
        Args:
            graph: KnowledgeGraph instance to process
            **kwargs: Additional processing parameters
        
        Returns:
            ModuleResult describing changes made
        """
        pass
    
    def configure(self, **kwargs):
        """Update module configuration."""
        self.config.update(kwargs)
    
    def validate(self) -> bool:
        """Validate module configuration."""
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get module information."""
        return {
            'name': self.name,
            'description': self.description,
            'config': self.config
        }
