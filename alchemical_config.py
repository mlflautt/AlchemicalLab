"""
Configuration Management for AlchemicalLab.

Loads and manages configuration from config.yaml with environment variable overrides.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """Configuration manager for AlchemicalLab."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.load()
    
    def load(self, config_path: str = None):
        """Load configuration from YAML file with environment overrides."""
        if config_path is None:
            config_path = os.environ.get('ALCHEMICAL_CONFIG', 'config.yaml')
        
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._get_defaults()
        
        self._apply_env_overrides()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'app': {
                'name': 'AlchemicalLab',
                'version': '0.1.0',
                'debug': False,
                'log_level': 'INFO',
            },
            'graph': {
                'db_path': 'knowledge_graph.db',
                'chroma_path': './chroma_db',
                'enable_vector_search': True,
            },
            'storylab': {
                'llm_provider': 'ollama',
                'model': 'phi3.5',
                'base_url': 'http://localhost:11434',
                'temperature': 0.7,
            },
            'calab': {
                'world_size': [100, 100],
                'default_density': 0.3,
                'random_seed': 42,
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
            },
        }
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'ALCHEMICAL_DB_PATH': ('graph', 'db_path'),
            'ALCHEMICAL_CHROMA_PATH': ('graph', 'chroma_path'),
            'ALCHEMICAL_LLM_PROVIDER': ('storylab', 'llm_provider'),
            'ALCHEMICAL_LLM_MODEL': ('storylab', 'model'),
            'ALCHEMICAL_LLM_URL': ('storylab', 'base_url'),
            'ALCHEMICAL_LLM_API_KEY': ('storylab', 'api_key'),
            'ALCHEMICAL_DEBUG': ('app', 'debug'),
            'ALCHEMICAL_LOG_LEVEL': ('app', 'log_level'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if section not in self._config:
                    self._config[section] = {}
                if key == 'debug':
                    value = value.lower() in ('true', '1', 'yes')
                elif key == 'port':
                    value = int(value)
                self._config[section][key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'graph.db_path')."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set config value using dot notation."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section."""
        return self._config.get(section, {})
    
    def save(self, config_path: str = None):
        """Save current config to YAML file."""
        if config_path is None:
            config_path = 'config.yaml'
        
        with open(config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def reload(self):
        """Reload configuration from file."""
        self.load()
    
    @property
    def graph_db_path(self) -> str:
        return self.get('graph.db_path', 'knowledge_graph.db')
    
    @property
    def graph_chroma_path(self) -> str:
        return self.get('graph.chroma_path', './chroma_db')
    
    @property
    def llm_provider(self) -> str:
        return self.get('storylab.llm_provider', 'ollama')
    
    @property
    def llm_model(self) -> str:
        return self.get('storylab.model', 'phi3.5')
    
    @property
    def llm_base_url(self) -> str:
        return self.get('storylab.base_url', 'http://localhost:11434')
    
    @property
    def llm_temperature(self) -> float:
        return self.get('storylab.temperature', 0.7)
    
    @property
    def debug(self) -> bool:
        return self.get('app.debug', False)
    
    @property
    def log_level(self) -> str:
        return self.get('app.log_level', 'INFO')


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()


def load_config(config_path: str = None) -> Config:
    """Load and return configuration."""
    config = Config()
    config.load(config_path)
    return config