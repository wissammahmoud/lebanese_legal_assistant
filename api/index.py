import os
import sys

# Add the root directory to the sys.path so 'app' can be found
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.main import app

# This is what Vercel will run