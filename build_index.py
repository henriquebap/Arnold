import json, gzip
from pathlib import Path
import rag_store as rs            # importa o módulo novo

HISTORY = Path("data/history.jsonl")

with HISTORY.open(encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        rs.add_record(rec)

print("Indexação concluída.")
