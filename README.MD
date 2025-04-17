## Passo a passo - Do zero ao LLM funcional local.
- (Testado em macOS14 + MacBook Pro M-series; ajuste paths se estiver no Linux/Windows)

### 1. Instalar pré-requisitos
(nao se esqueca de configurar o ambiente virtual)
```
# 1.1 homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 1.2 Build
brew install git cmake wget python@3.11

````
### 2. Instalar e inicial o Ollama
```
brew install ollama  #instala o binario universal
ollama serve  #inicia o servidor (deixe em seugndo plano - porta 11434)
```

### 3. Baixar o modelo Mistral-7b
```
ollama pull mistral:7b  #base, mistral:7b-instruct Variante de chat "instruct"
```
verifique :
```
ollama list
# NAME               ID            SIZE       MODIFIED
# mistral:7b...                   4.1 GB        now
```
teste:
````
ollama run mistral:7b
# /bye para sair
`````

### 4. Criar o projeto python
````
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install requests

````

### 5. Salvar 'coach.py'
`````
coach.py
....
`````

### 6. Executar
````
python coach.py
```
Output será algo:
````
---
  Here is your customized leg and core workout for Monday, Henrique. Remember to always warm up before starting and cool down afterwards.

**Exercise 1: Back Squat (Lower Body Strength)**
- Sets: 4
- Reps: 8
- Safety Cue: Keep your heels on the ground throughout the squat.

**Exercise 2: Goblet Squat (Lower Body Endurance)**
- Sets: 3
- Reps: 12
- Safety Cue: Maintain a neutral spine and engage your core.

**Exercise 3: Romanian Deadlift (Back Strength)**
- Sets: 3
- Reps: 8
- Safety Cue: Keep a flat back during the entire movement.

**Exercise 4: Hip Thrusts (Lower Body Strength & Core Endurance)**
- Sets: 3
- Reps: 12
- Safety Cue: Push your heels firmly into the ground and squeeze your glutes at the top of each rep.

**Core Exercise 1: Plank (Core Stability)**
- Sets: 3
- Reps: Hold for 45 seconds
- Safety Cue: Keep a neutral spine and engage your core throughout the movement.

**Core Exercise 2: Russian Twists (Core Rotation & Flexibility)**
- Sets: 3
- Reps: 15 each side
- Safety Cue: Keep your back flat and twist from your torso, not your shoulders.

---

To ensure you're maximizing the benefits of this workout, consider focusing on controlled movements and proper form throughout each exercise. As you progress, adjust the sets, reps, or weights accordingly.

**Question for Refining Next Session:** What days of the week are you typically available to work out, so we can plan a consistent training schedule tailored to your climbing goals?
````

