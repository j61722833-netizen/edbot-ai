#!/usr/bin/env python3
"""
EdBot AI - Secure API Key Setup Utility

Interactive tool for securely configuring API keys with multiple storage options:
1. System keyring (most secure for local development)
2. .env file (convenient for development)
3. Environment variables (production)

Security features:
- Hidden input for API keys
- Format validation
- Secure keyring storage
- Clear setup instructions
"""

import sys
import os
from pathlib import Path
from getpass import getpass
import json

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.config.settings import get_config, reload_config
except ImportError as e:
    print(f"❌ Error importing configuration: {e}")
    print("Please ensure dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False


class APIKeySetup:
    """Interactive API key setup utility."""
    
    def __init__(self):
        self.config = get_config()
        
    def run_interactive_setup(self):
        """Run the interactive setup process."""
        print("🔐 EdBot AI - Secure API Key Setup")
        print("=" * 50)
        
        # Check current status
        current_status = self.config.get_config_summary()
        
        print("📊 Current Configuration Status:")
        print(f"  OpenAI API Key: {'✅ Set' if current_status['openai_api_key_set'] else '❌ Missing'}")
        print(f"  LangSmith API Key: {'✅ Set' if current_status['langsmith_api_key_set'] else '❌ Not set'} (optional)")
        print(f"  Serper API Key: {'✅ Set' if current_status['serper_api_key_set'] else '❌ Not set'} (optional)")
        print()
        
        # Show available storage options
        print("🗂 Available Storage Options:")
        print("1. System Keyring - Most secure (recommended)")
        if KEYRING_AVAILABLE:
            print("   ✅ Available")
        else:
            print("   ❌ Not available (install: pip install keyring)")
        
        print("2. .env file - Convenient for development")
        print("   ✅ Available")
        print("3. Environment variables - Production use")
        print("   ✅ Available")
        print()
        
        # Main setup menu
        while True:
            print("🛠 Setup Options:")
            print("1. Set up OpenAI API key (required)")
            print("2. Set up LangSmith API key (optional)")
            print("3. Set up Serper API key (optional)")
            print("4. Save configuration to .env file")
            print("5. Show current status")
            print("6. Clear keyring data")
            print("0. Exit")
            
            choice = input("\nSelect option (0-6): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                self.setup_openai_key()
            elif choice == "2":
                self.setup_langsmith_key()
            elif choice == "3":
                self.setup_serper_key()
            elif choice == "4":
                self.save_to_env_file()
            elif choice == "5":
                self.show_status()
            elif choice == "6":
                self.clear_keyring()
            else:
                print("❌ Invalid option. Please try again.")
            
            print()
    
    def setup_openai_key(self):
        """Setup OpenAI API key."""
        print("\n🤖 OpenAI API Key Setup")
        print("-" * 30)
        
        if not self.config.openai_api_key:
            print("📖 Get your API key from: https://platform.openai.com/api-keys")
        else:
            print("Current key is set. This will replace it.")
        
        print("\nStorage options:")
        if KEYRING_AVAILABLE:
            print("1. System keyring (secure)")
        print("2. .env file (convenient)")
        print("0. Cancel")
        
        choice = input("Choose storage method: ").strip()
        
        if choice == "0":
            return
        elif choice == "1" and KEYRING_AVAILABLE:
            self.store_key_in_keyring("openai", "OpenAI")
        elif choice == "2":
            self.store_key_for_env("openai", "OpenAI")
        else:
            print("❌ Invalid choice.")
    
    def setup_langsmith_key(self):
        """Setup LangSmith API key."""
        print("\n📊 LangSmith API Key Setup (Optional)")
        print("-" * 40)
        print("LangSmith enables tracing and monitoring of your AI applications.")
        print("📖 Get your key from: https://smith.langchain.com/")
        
        if input("Set up LangSmith? (y/n): ").lower().strip() in ('y', 'yes'):
            print("\nStorage options:")
            if KEYRING_AVAILABLE:
                print("1. System keyring (secure)")
            print("2. .env file (convenient)")
            print("0. Cancel")
            
            choice = input("Choose storage method: ").strip()
            
            if choice == "1" and KEYRING_AVAILABLE:
                self.store_key_in_keyring("langsmith", "LangSmith")
            elif choice == "2":
                self.store_key_for_env("langsmith", "LangSmith")
    
    def setup_serper_key(self):
        """Setup Serper API key."""
        print("\n🔍 Serper API Key Setup (Optional)")
        print("-" * 35)
        print("Serper enables web search for fact-checking capabilities.")
        print("📖 Get your key from: https://serper.dev/")
        
        if input("Set up Serper? (y/n): ").lower().strip() in ('y', 'yes'):
            print("\nStorage options:")
            if KEYRING_AVAILABLE:
                print("1. System keyring (secure)")
            print("2. .env file (convenient)")
            print("0. Cancel")
            
            choice = input("Choose storage method: ").strip()
            
            if choice == "1" and KEYRING_AVAILABLE:
                self.store_key_in_keyring("serper", "Serper")
            elif choice == "2":
                self.store_key_for_env("serper", "Serper")
    
    def store_key_in_keyring(self, service: str, display_name: str):
        """Store API key in system keyring."""
        if not KEYRING_AVAILABLE:
            print("❌ Keyring not available.")
            return
        
        key = getpass(f"Enter your {display_name} API key (hidden): ").strip()
        
        if not key:
            print("❌ No key provided.")
            return
        
        if service == "openai" and not self.validate_openai_key(key):
            print("❌ Invalid OpenAI API key format.")
            return
        
        try:
            keyring.set_password("edbot-ai", f"{service}_api_key", key)
            print(f"✅ {display_name} API key stored securely in keyring!")
            
            # Reload config to pick up new key
            reload_config()
            
        except Exception as e:
            print(f"❌ Error storing key: {e}")
    
    def store_key_for_env(self, service: str, display_name: str):
        """Store API key for .env file."""
        key = getpass(f"Enter your {display_name} API key (hidden): ").strip()
        
        if not key:
            print("❌ No key provided.")
            return
        
        if service == "openai" and not self.validate_openai_key(key):
            print("❌ Invalid OpenAI API key format.")
            return
        
        # Store in memory for saving to .env later
        key_name = f"{service.upper()}_API_KEY"
        os.environ[key_name] = key
        
        print(f"✅ {display_name} API key ready for .env file!")
        print("💡 Use option 4 to save all keys to .env file.")
    
    def validate_openai_key(self, key: str) -> bool:
        """Validate OpenAI API key format."""
        return bool(key and key.startswith("sk-") and len(key) > 20)
    
    def save_to_env_file(self):
        """Save configuration to .env file."""
        print("\n💾 Save Configuration to .env File")
        print("-" * 35)
        
        env_file = Path(".env")
        if env_file.exists():
            print("⚠️  .env file already exists.")
            if input("Overwrite? (y/n): ").lower().strip() not in ('y', 'yes'):
                return
        
        try:
            # Reload config to get current state
            config = reload_config()
            saved_path = config.save_to_env_file(include_comments=True)
            
            print(f"✅ Configuration saved to: {saved_path}")
            print("🔒 Remember: .env file is excluded from git (secure)")
            print("📖 You can edit the .env file manually if needed")
            
        except Exception as e:
            print(f"❌ Error saving .env file: {e}")
    
    def show_status(self):
        """Show detailed configuration status."""
        print("\n📊 Configuration Status")
        print("-" * 25)
        
        config = reload_config()
        status = config.get_config_summary()
        
        print(f"System configured: {'✅ Yes' if status['configured'] else '❌ No'}")
        print("\nAPI Keys:")
        print(f"  OpenAI: {'✅ Set' if status['openai_api_key_set'] else '❌ Missing (required)'}")
        print(f"  LangSmith: {'✅ Set' if status['langsmith_api_key_set'] else '❌ Not set (optional)'}")
        print(f"  Serper: {'✅ Set' if status['serper_api_key_set'] else '❌ Not set (optional)'}")
        
        print(f"\nModels:")
        print(f"  Embeddings: {status['embeddings_model']}")
        print(f"  Q&A: {status['qa_model']}")
        
        print(f"\nFeatures:")
        print(f"  LangSmith tracing: {'✅ Enabled' if status['langsmith_tracing'] else '❌ Disabled'}")
        print(f"  Keyring support: {'✅ Available' if status['keyring_available'] else '❌ Not available'}")
        print(f"  .env support: {'✅ Available' if status['dotenv_available'] else '❌ Not available'}")
        
        if status.get('config_sources'):
            print(f"\nConfiguration sources:")
            for key, source in status['config_sources'].items():
                print(f"  {key}: {source}")
    
    def clear_keyring(self):
        """Clear API keys from keyring."""
        if not KEYRING_AVAILABLE:
            print("❌ Keyring not available.")
            return
        
        print("\n🗑 Clear Keyring Data")
        print("-" * 20)
        print("This will remove all API keys from your system keyring.")
        
        if input("Are you sure? (y/n): ").lower().strip() in ('y', 'yes'):
            try:
                config = get_config()
                config.clear_keyring_data()
            except Exception as e:
                print(f"❌ Error clearing keyring: {e}")


def main():
    """Main setup function."""
    try:
        setup = APIKeySetup()
        setup.run_interactive_setup()
        
        print("\n✨ Setup Complete!")
        print("🚀 You can now use EdBot AI:")
        print("   ./run.sh list")
        print("   ./run.sh process textbook.pdf")
        print("   ./run.sh chat textbook_name")
        
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()