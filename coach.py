import requests, json, textwrap

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


def ask_llm(message: str, profile: dict, stream: bool = False) -> str:
    """Returns the full assistant response as a string."""
    payload = {
        "model": "mistral:7b",
        "prompt": SYSTEM.format(**profile) +
                  f"\n\nUser: {message}\nAssistant:",
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


if __name__ == "__main__":
    profile = {
        "name": "Henrique",
        "age": 21,
        "height_cm": 173,
        "weight_kg": 73,
        "goals": "climbing endurance, leg & back strength",
        "constraints": "no injuries, full commercial gym"
    }

    # → retorna string (sem generator)
    reply = ask_llm("Plan my Monday leg/core session.", profile, stream=False)
    print("\n---\n", reply)



# def log_interaction(user_msg, assistant_msg, profile):
#     with open("data/history.jsonl", "a") as f:
#         f.write(json.dumps({
#             "ts": datetime.now().isoformat(),
#             "profile": profile,
#             "user": user_msg,
#             "assistant": assistant_msg
#         }) + "\n")
