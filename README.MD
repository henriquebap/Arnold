## Passo a passo - Do zero ao LLM funcional local
(Testado em macOS 14 + MacBook Pro M-series; ajuste paths se estiver no Linux/Windows)

---

### 1. Instalar pré-requisitos
*(não se esqueça de configurar o ambiente virtual)*

```bash
# 1.1 Instalar Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 1.2 Instalar ferramentas de build
brew install git cmake wget python@3.11
```

---

### 2. Instalar e iniciar o Ollama

```bash
brew install ollama  # instala o binário universal
ollama serve         # inicia o servidor (deixe rodando em segundo plano - porta 11434)
```

---

### 3. Baixar o modelo Mistral-7b

```bash
ollama pull mistral:7b  # base model, ou mistral:7b-instruct para a variante de chat "instruct"
```

Verificar:

```bash
ollama list
# NAME               ID            SIZE       MODIFIED
# mistral:7b...                   4.1 GB        now
```

Testar:

```bash
ollama run mistral:7b
# use /bye para sair
```

---

### 4. Criar o projeto Python

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install requests
```

---

### 5. Salvar o arquivo `coach.py`

```text
coach.py
....
```

---

### 6. Executar

```bash
python coach.py
```

O output será algo como:

```text
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
```

---

## English Version

## Step-by-Step - From Zero to a Functional Local LLM
(Tested on macOS 14 + MacBook Pro M-series; adjust paths if on Linux/Windows)

---

### 1. Install prerequisites
*(don't forget to set up the virtual environment)*

```bash
# 1.1 Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 1.2 Install build tools
brew install git cmake wget python@3.11
```

---

### 2. Install and start Ollama

```bash
brew install ollama  # installs the universal binary
ollama serve         # starts the server (leave it running in the background - port 11434)
```

---

### 3. Download the Mistral-7b model

```bash
ollama pull mistral:7b  # base model, or mistral:7b-instruct for the chat "instruct" variant
```

Check:

```bash
ollama list
# NAME               ID            SIZE       MODIFIED
# mistral:7b...                   4.1 GB        now
```

Test:

```bash
ollama run mistral:7b
# use /bye to exit
```

---

### 4. Create the Python project

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install requests
```

---

### 5. Save the `coach.py` file

```text
coach.py
....
```

---

### 6. Run

```bash
python coach.py
```

The output will look like:

```text
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
```

