import requests, json, textwrap, uuid
from datetime import datetime
from pathlib import Path
import rag_store as rs

OLLAMA = "http://localhost:11434/api/generate"

SYSTEM = textwrap.dedent("""
You are GymCoachGPT, an evidence-based strength & conditioning coach.
Rules:
• tailor everything to the USER PROFILE
• answer in markdown
• add sets x reps and 1 safety cue per exercise
• finish with one question that helps refine next session
== USER PROFILE ==
{name}, {age} y -{weight_kg} kg / {height_cm} cm
Goals: {goals}
Constraints: {constraints}
""").strip()

LOG_FILE = Path("data/history.jsonl"); LOG_FILE.parent.mkdir(exist_ok=True)

def build_prompt(message, profile):
    memories = rs.query_memories(message, k=3)
    memory_block = "\n\n== RELEVANT PAST INTERACTIONS ==\n" + "\n---\n".join(memories) if memories else ""
    return SYSTEM.format(**profile) + memory_block + f"\n\nUser: {message}\nAssistant:"

def ask_llm(message: str, profile: dict, stream: bool = False) -> str:
    """Returns the full assistant response as a string."""
    prompt = build_prompt(message, profile)
    payload = {
        "model": "mistral:7b",
        "prompt": prompt,
        "stream": stream
    }

    if stream:
        # Mantém a conexão aberta para receber pedaços (“server‑sent events”)
        with requests.post(OLLAMA, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            full_text = ""
            for line in r.iter_lines():
                if not line:
                    continue
                # cada linha começa com b'data: '
                chunk = json.loads(line[6:])["response"]
                full_text += chunk
                print(chunk, end="", flush=True)  # opcional: ver em tempo real
        print()  # quebra de linha no fim
        return full_text

    # modo não‑stream: resposta já vem completa em JSON
    r = requests.post(OLLAMA, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]


def log_and_index(user, assistant_msg, profile):
    rec = {
        "id": str(uuid.uuid4()),
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "user": user,
        "assistant": assistant_msg,
        "profile": profile
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    rs.add_record(rec)


if __name__ == "__main__":
    profile = {
        "name": "Henrique",
        "age": 21,
        "height_cm": 173,
        "weight_kg": 73,
        "goals": "climbing endurance, leg & back strength",
        "constraints": "no injuries, full commercial gym"
    }

    msg = "Plan my Friday Full Body + grip session, I'm kind tired from the last workout."
    reply = ask_llm(msg, profile, stream=False)
    print("\n---\n", reply)

    log_and_index(msg, reply, profile)


