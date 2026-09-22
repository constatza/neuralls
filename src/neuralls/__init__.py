"""neuralls library modules.

This package contains the core library functionality for the neuralls project,
separated from the application scripts for better organization and reusability.
"""

import os

# mlflow's agent-hint banner (mlflow/agent/hint.py) points coding agents at its
# GenAI/LLM tracing skill on every import; irrelevant here (this project uses
# plain MLflow experiment tracking, no spans/tracing) and noisy in every agent
# session. Set before any submodule imports mlflow.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
