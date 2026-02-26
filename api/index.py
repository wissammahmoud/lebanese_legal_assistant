import sys
import os

# Adds the root directory to the path so 'app' is discoverable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app