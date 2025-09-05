"""
Secure configuration management for EdBot AI.

Handles API keys and sensitive configuration with multiple fallback methods:
1. Environment variables (production)
2. .env file (development) 
3. System keyring (secure local storage)
4. Interactive prompts (setup)

Security features:
- Never logs or prints API keys
- Validates API key format
- Secure keyring storage option
- Clear separation of sensitive/non-sensitive config
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from getpass import getpass

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

@dataclass
class EdBotConfig:
    """Secure configuration for EdBot AI system."""
    
    # API Configuration (sensitive)
    openai_api_key: Optional[str] = field(default=None, repr=False)
    langsmith_api_key: Optional[str] = field(default=None, repr=False)
    serper_api_key: Optional[str] = field(default=None, repr=False)
    
    # LangSmith Configuration
    langsmith_tracing: bool = False
    langsmith_project: str = "edbot-ai-textbook-qa"
    
    # Model Configuration
    embeddings_model: str = "text-embedding-3-large"
    qa_model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    # Vector Database Configuration
    faiss_index_path: str = "./vector_indexes"
    
    # Processing Configuration
    pages_per_chunk: int = 15
    min_text_length: int = 50
    max_chunk_size: int = 1000
    max_context_chunks: int = 5
    min_relevance_score: float = 0.7
    
    # File paths
    output_dir: str = "./outputs"
    
    # Security settings
    _keyring_service: str = field(default="edbot-ai", init=False, repr=False)
    _config_sources: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self):
        """Load configuration from various sources."""
        self._load_configuration()
        self._validate_configuration()
    
    def _load_configuration(self):
        """Load configuration from multiple sources in order of preference."""
        # 1. Load from .env file if available
        if DOTENV_AVAILABLE:
            env_file = PROJECT_ROOT / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                self._config_sources[".env"] = str(env_file)
                logger.debug("Loaded configuration from .env file")
        
        # 2. Load from environment variables
        self._load_from_environment()
        
        # 3. Load from system keyring if available
        if KEYRING_AVAILABLE:
            self._load_from_keyring()
        
        # 4. Validate required keys
        if not self.openai_api_key:
            self._handle_missing_openai_key()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # API keys
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if self.openai_api_key:
                self._config_sources["openai_api_key"] = "environment"
        
        if not self.langsmith_api_key:
            self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
            if self.langsmith_api_key:
                self._config_sources["langsmith_api_key"] = "environment"
        
        if not self.serper_api_key:
            self.serper_api_key = os.getenv("SERPER_API_KEY")
            if self.serper_api_key:
                self._config_sources["serper_api_key"] = "environment"
        
        # LangSmith settings
        langsmith_tracing = os.getenv("LANGSMITH_TRACING", "").lower()
        if langsmith_tracing in ("true", "1", "yes", "on"):
            self.langsmith_tracing = True
        
        # Model settings
        self.embeddings_model = os.getenv("EMBEDDINGS_MODEL", self.embeddings_model)
        self.qa_model = os.getenv("QA_MODEL", self.qa_model)
        
        # Paths
        self.faiss_index_path = os.getenv("FAISS_INDEX_PATH", self.faiss_index_path)
        self.output_dir = os.getenv("OUTPUT_DIR", self.output_dir)
    
    def _load_from_keyring(self):
        """Load API keys from system keyring."""
        try:
            if not self.openai_api_key:
                key = keyring.get_password(self._keyring_service, "openai_api_key")
                if key:
                    self.openai_api_key = key
                    self._config_sources["openai_api_key"] = "keyring"
            
            if not self.langsmith_api_key:
                key = keyring.get_password(self._keyring_service, "langsmith_api_key")
                if key:
                    self.langsmith_api_key = key
                    self._config_sources["langsmith_api_key"] = "keyring"
            
            if not self.serper_api_key:
                key = keyring.get_password(self._keyring_service, "serper_api_key")
                if key:
                    self.serper_api_key = key
                    self._config_sources["serper_api_key"] = "keyring"
                    
        except Exception as e:
            logger.debug(f"Could not load from keyring: {e}")
    
    def _handle_missing_openai_key(self):
        """Handle missing OpenAI API key with various options."""
        if os.getenv("EDBOT_NON_INTERACTIVE"):
            # Non-interactive mode (CI/CD, testing)
            logger.warning("OpenAI API key not found. Running in non-interactive mode.")
            return
        
        print("\n🔑 OpenAI API Key Required")
        print("=" * 40)
        print("EdBot AI requires an OpenAI API key to function.")
        print("\nOptions to configure:")
        print("1. Environment variable: export OPENAI_API_KEY='your-key'")
        print("2. .env file: Create .env with OPENAI_API_KEY=your-key")
        
        if KEYRING_AVAILABLE:
            print("3. System keyring: Secure local storage")
        
        print("\n" + "=" * 40)
        
        # Offer to set up keyring storage
        if KEYRING_AVAILABLE:
            response = input("Would you like to securely store your API key? (y/n): ").lower().strip()
            if response in ('y', 'yes'):
                self._setup_keyring_storage()
        
        # If still no key, show setup instructions
        if not self.openai_api_key:
            print("\n📖 Setup Instructions:")
            print("1. Get your API key from: https://platform.openai.com/api-keys")
            print("2. Run: export OPENAI_API_KEY='your-key-here'")
            print("3. Or create .env file with: OPENAI_API_KEY=your-key-here")
    
    def _setup_keyring_storage(self):
        """Interactive setup for keyring storage."""
        try:
            print("\n🔐 Secure API Key Setup")
            print("Your API key will be stored securely in your system's keyring.")
            
            api_key = getpass("Enter your OpenAI API key (input hidden): ").strip()
            
            if api_key and self._validate_openai_key_format(api_key):
                keyring.set_password(self._keyring_service, "openai_api_key", api_key)
                self.openai_api_key = api_key
                self._config_sources["openai_api_key"] = "keyring"
                print("✅ API key stored securely in keyring!")
            else:
                print("❌ Invalid API key format. Please try again.")
                
        except Exception as e:
            print(f"❌ Error setting up keyring: {e}")
    
    def _validate_openai_key_format(self, key: str) -> bool:
        """Validate OpenAI API key format."""
        return bool(key and key.startswith("sk-") and len(key) > 20)
    
    def _validate_configuration(self):
        """Validate loaded configuration."""
        # Validate OpenAI API key format
        if self.openai_api_key and not self._validate_openai_key_format(self.openai_api_key):
            logger.warning("OpenAI API key format appears invalid")
        
        # Validate numeric ranges
        if not (0 <= self.temperature <= 2):
            raise ValueError("Temperature must be between 0 and 2")
        
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        
        if not (0 <= self.min_relevance_score <= 1):
            raise ValueError("min_relevance_score must be between 0 and 1")
    
    def is_configured(self) -> bool:
        """Check if minimum required configuration is present."""
        return bool(self.openai_api_key)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get non-sensitive configuration summary."""
        return {
            "configured": self.is_configured(),
            "openai_api_key_set": bool(self.openai_api_key),
            "langsmith_api_key_set": bool(self.langsmith_api_key),
            "serper_api_key_set": bool(self.serper_api_key),
            "embeddings_model": self.embeddings_model,
            "qa_model": self.qa_model,
            "langsmith_tracing": self.langsmith_tracing,
            "config_sources": {k: v for k, v in self._config_sources.items() 
                             if not k.endswith("_api_key")},
            "keyring_available": KEYRING_AVAILABLE,
            "dotenv_available": DOTENV_AVAILABLE
        }
    
    def save_to_env_file(self, include_comments: bool = True) -> Path:
        """Save current configuration to .env file (excluding sensitive keys from keyring)."""
        env_file = PROJECT_ROOT / ".env"
        
        content = []
        
        if include_comments:
            content.extend([
                "# EdBot AI Configuration",
                "# Secure API key storage for textbook Q&A system",
                "# Generated automatically - do not commit to version control",
                "",
                "# OpenAI Configuration (Required)",
            ])
        
        # Only include API keys that aren't from keyring
        if self.openai_api_key and self._config_sources.get("openai_api_key") != "keyring":
            content.append(f"OPENAI_API_KEY={self.openai_api_key}")
        elif include_comments:
            content.append("# OPENAI_API_KEY=your_openai_api_key_here")
        
        if include_comments:
            content.extend(["", "# LangSmith Configuration (Optional)"])
        
        if self.langsmith_api_key and self._config_sources.get("langsmith_api_key") != "keyring":
            content.append(f"LANGSMITH_API_KEY={self.langsmith_api_key}")
        elif include_comments:
            content.append("# LANGSMITH_API_KEY=your_langsmith_key_here")
        
        content.extend([
            f"LANGSMITH_TRACING={'true' if self.langsmith_tracing else 'false'}",
            f"LANGSMITH_PROJECT={self.langsmith_project}"
        ])
        
        if include_comments:
            content.extend(["", "# Model Configuration"])
        
        content.extend([
            f"EMBEDDINGS_MODEL={self.embeddings_model}",
            f"QA_MODEL={self.qa_model}",
            f"TEMPERATURE={self.temperature}",
            f"MAX_TOKENS={self.max_tokens}"
        ])
        
        if include_comments:
            content.extend(["", "# Storage Configuration"])
        
        content.extend([
            f"FAISS_INDEX_PATH={self.faiss_index_path}",
            f"OUTPUT_DIR={self.output_dir}"
        ])
        
        if self.serper_api_key and self._config_sources.get("serper_api_key") != "keyring":
            if include_comments:
                content.extend(["", "# External Search (Optional)"])
            content.append(f"SERPER_API_KEY={self.serper_api_key}")
        
        # Write to file
        with open(env_file, 'w') as f:
            f.write('\n'.join(content) + '\n')
        
        return env_file
    
    def clear_keyring_data(self):
        """Clear API keys from system keyring."""
        if not KEYRING_AVAILABLE:
            return
        
        try:
            keyring.delete_password(self._keyring_service, "openai_api_key")
            keyring.delete_password(self._keyring_service, "langsmith_api_key") 
            keyring.delete_password(self._keyring_service, "serper_api_key")
            print("✅ Cleared API keys from keyring")
        except Exception as e:
            logger.debug(f"Error clearing keyring: {e}")


# Global configuration instance
_config: Optional[EdBotConfig] = None

def get_config() -> EdBotConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = EdBotConfig()
    return _config

def reload_config() -> EdBotConfig:
    """Reload configuration from sources."""
    global _config
    _config = EdBotConfig()
    return _config

def is_configured() -> bool:
    """Quick check if system is properly configured."""
    return get_config().is_configured()


if __name__ == "__main__":
    # CLI for configuration management
    import argparse
    
    parser = argparse.ArgumentParser(description="EdBot AI Configuration Management")
    parser.add_argument("--status", action="store_true", help="Show configuration status")
    parser.add_argument("--setup", action="store_true", help="Interactive setup")
    parser.add_argument("--save-env", action="store_true", help="Save to .env file")
    parser.add_argument("--clear-keyring", action="store_true", help="Clear keyring data")
    
    args = parser.parse_args()
    
    config = get_config()
    
    if args.status:
        import json
        summary = config.get_config_summary()
        print("📊 EdBot AI Configuration Status")
        print("=" * 40)
        print(json.dumps(summary, indent=2))
    
    elif args.setup:
        print("🛠 EdBot AI Setup")
        if not config.is_configured():
            config._handle_missing_openai_key()
        else:
            print("✅ System already configured!")
    
    elif args.save_env:
        env_file = config.save_to_env_file()
        print(f"💾 Configuration saved to: {env_file}")
    
    elif args.clear_keyring:
        config.clear_keyring_data()
    
    else:
        parser.print_help()