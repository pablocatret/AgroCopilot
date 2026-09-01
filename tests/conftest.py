import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_EXTERNALS", "1")
os.environ.setdefault("SSE_LOG_MODE", "memory")
os.environ.setdefault("OCR_BACKEND", "none")
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENROUTER_API_KEY", "")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
