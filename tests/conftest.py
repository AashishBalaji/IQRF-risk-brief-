import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the real SQLite query log at a throwaway temp file for the
# whole test session, so running the suite never writes
# rag_query_log.db into the repo itself (api.py's lifespan opens this
# at TestClient startup -- see rag/query_log.py).
os.environ.setdefault(
    "IQRF_QUERY_LOG_PATH", str(Path(tempfile.gettempdir()) / "iqrf_test_query_log.db")
)
