"""
Configuration loader for Customer Service Chatbot
Loads API keys from config/secrets.json
"""

import json
import os
from pathlib import Path

def load_config():
    """Load API configuration from secrets.json"""
    config_path = Path(__file__).parent / "config" / "secrets.json"
    
    if not config_path.exists():
        print("⚠️  Warning: config/secrets.json not found!")
        print("📝 Please copy config/secrets.example.json to config/secrets.json")
        print("   and add your OpenAI API key")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY", "")
    
    return config

def get_model_name(config=None):
    """Get the model name from config or use default"""
    if config is None:
        config = load_config()
    
    if config:
        return config.get("MODEL_NAME", "gpt-4o-mini")
    else:
        return "gpt-4o-mini"
