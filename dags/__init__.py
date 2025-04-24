"""
DAGs package for ML Automation.

This package contains all DAGs and their supporting modules.
"""

import os
import sys

# Ensure the correct Python paths are set up 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add paths if they don't exist
for path in [current_dir, parent_dir]:
    if path not in sys.path:
        sys.path.append(path)

__version__ = "1.0.0" 