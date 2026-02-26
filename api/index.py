import sys
import os

# This tells Python to look in the root directory for the 'app' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app