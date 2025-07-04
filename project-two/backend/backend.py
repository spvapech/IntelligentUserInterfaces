from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import ollama

app = FastAPI(title="LLM Writing Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# System-Prompt für alle Anfragen
SYSTEM_PROMPT = (
    "You are an expert academic writing assistant. "
    "Improve the given text according to the user's instruction."
)

def make_prompt(text: str, mode: str, weight: float) -> str:
    # Der "weight"-Parameter wird hier primär für die temperature im Modell genutzt.
    if mode == "grammar":
        return f"Bitte korrigiere die Grammatik des folgenden Textes (Intensität {weight}). Konzentriere dich nur auf Grammatik und Rechtschreibung und gib mir NUR die Antwort, kommentiere sie nicht:\n\n'''{text}'''"
    elif mode == "Paraphrasing":
        return f"Bitte paraphrasiere den folgenden Text (Intensität {weight}), um dieselbe Bedeutung mit anderen Worten auszudrücken, und gib mir NUR die Antwort, kommentiere sie nicht:\n\n'''{text}'''"
    elif mode == "Proofreading":
        return f"Bitte korrigiere den folgenden Text (Intensität {weight}) auf Fehler in Grammatik, Rechtschreibung, Zeichensetzung und Stil. Schlage Verbesserungen für Klarheit und Prägnanz vor und gib mir NUR die Antwort, kommentiere sie nicht:\n\n'''{text}'''"
    elif mode == "Citation":
        return f"Bitte überprüfe den folgenden Text (Intensität {weight}) und schlage vor, wie Informationen, die eine Quellenangabe erfordern, in einem akademischen Kontext korrekt zitiert werden können. Gib möglichst Beispielzitate an und gib mir NUR die Antwort, kommentiere sie nicht:\n\n'''{text}'''"
    elif mode == "Summarizer":
        return f"Bitte fasse den folgenden Text (Intensität {weight}) prägnant und genau zusammen und gib mir NUR die Antwort, kommentiere sie nicht:\n\n'''{text}'''"
    else: # Standardmodus "full"
        return f"Bitte überarbeite den folgenden Text (Intensität {weight}) umfassend, um seine Qualität, Klarheit und den akademischen Ton zu verbessern, und gib mir NUR die Antwort, kommentiere sie nicht:\n\n'''{text}'''"

OLLAMA_MODEL_NAME = "phi3:mini" # <--


@app.post("/assist")
async def assist_endpoint(request: Request):
    data = await request.json()
    original_text = data.get("text", "")
    mode          = data.get("mode", "full")
    weight        = data.get("weight", 0.5) # Wird als 'temperature' verwendet

    if not original_text:
        return {"assisted_text": ""}

    prompt = make_prompt(original_text, mode, weight)

    try:
        # 1) chat() sendet Messages an Ollama REST API

        response = ollama.chat(
            model="llama3.2:latest", # <-- Dein Ollama-Modell hier verwenden
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": prompt}
            ],
            options={
                "temperature": weight # 'weight' als temperature übergeben
            }
        )

        # 2) Die Antwort vom Modell:
        assisted = response.message.content

    except Exception as e:
        print(f"Fehler bei der Ollama API-Anfrage: {e}")
        assisted = f"Entschuldigung, ein Fehler ist bei der Verbindung mit dem lokalen Ollama-Server aufgetreten: {e}. Bitte überprüfe, ob die Ollama-App läuft und das Modell {OLLAMA_MODEL_NAME} heruntergeladen ist."

    return {"assisted_text": assisted}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))