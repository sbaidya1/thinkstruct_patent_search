"""
Application factory for the Flask app.

- Loads environment variables from .env file
- Sets up Flask configuration (secret key, API keys, chroma path)
- Registers route blueprints
"""

import os
from flask import Flask
from dotenv import load_dotenv

# load env variables from .env file
load_dotenv()

def create_app():
  app = Flask(__name__)

  # base config values 
  app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-fallback')
  app.config['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
  app.config["CHROMA_PERSIST_DIR"] = "chroma_index"

  # register blueprints
  from .routes import main
  app.register_blueprint(main)

  return app
