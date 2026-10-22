from dotenv import load_dotenv
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)

print("Project root:", PROJECT_ROOT)
print(".env exists:", (PROJECT_ROOT / ".env").exists())
print("Environment loaded:", bool(os.getenv("LLM_API_KEY")))
