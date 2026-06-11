"""conftest.py -- Shared fixtures for SARL-on-MARL tests."""

import os
import sys

# Ensure project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
