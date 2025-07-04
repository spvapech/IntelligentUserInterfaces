import streamlit as st
import requests
import difflib
from typing import Dict, Any

# Seite konfigurieren
st.set_page_config(page_title="LLM Writing Assistant", layout="wide")

# Farben laut Figma
BG_COLOR = "#393E41"
CARD_BG = "#4D4F50"
TEXT_COLOR = "#FFFFFF"
BUTTON_COLOR = "#D9D9D9"
BUTTON_TEXT_COLOR = "#1E1E1E"

API_URL = "http://127.0.0.1:8000"

# CSS-Styling
st.markdown("""
    <style>
    body, .stApp {
        background-color: #393E41;
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Hauptcontainer für TextArea (inkl. äußeren Fokusrahmen) */
    div[data-baseweb="textarea"], .stTextArea > div {
        background-color: #4D4F50 !important;
        border-radius: 40px !important;
        border: none !important;
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.3) !important;
        outline: none !important;
    }

    /* Wenn Fokus: keine rote Linie anzeigen */
    div[data-baseweb="textarea"]:focus-within {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Textarea selbst */
    textarea {
        background-color: #4D4F50 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 40px !important;
        padding-top: 100px !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
        text-align: center !important;
        font-size: 20px !important;
        line-height: 1.6 !important;
        resize: none !important;
        height: 320px !important;
        outline: none !important;
        box-shadow: none !important;
    }

    textarea:focus {
        outline: none !important;
        border: none !important;
        box-shadow: none !important;
    }

    button[kind="secondary"] {
        background-color: #D9D9D9 !important;
        color: #1E1E1E !important;
        border: none !important;
        border-radius: 32px !important;
        height: 80px !important;
        width: 220px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
        transition: background-color 0.3s ease-in-out, transform 0.2s ease-in-out;
    }

    button[kind="secondary"]:hover {
        background-color: #cfcfcf !important;
        transform: translateY(-2px) !important;
    }
    </style>
""", unsafe_allow_html=True)


# Backend-Request
def send_text_for_assistance(text: str, mode: str, weight: float) -> Dict[str, Any]:
    payload = {"text": text, "mode": mode, "weight": weight}
    try:
        response = requests.post(f"{API_URL}/assist", json=payload)
        return response.json()
    except requests.RequestException as e:
        st.error(f"Fehler beim Verbinden mit dem Backend: {e}")
        return {"assisted_text": ""}

# Diff-Funktion
def display_diff(original: str, revised: str) -> str:
    """
    Generate a diff view between original and revised text.

    This is a simple implementation. Students should enhance this with
    better formatting, color-coding, or a more sophisticated diff library.

    Parameters:
        original (str): The original text
        revised (str): The revised text

    Returns:
        str: The diff text showing changes
    """
    diff = difflib.ndiff(original.splitlines(), revised.splitlines())
    return "\n".join(diff)

# Haupt-App
def main():
    st.markdown("<h2 style='text-align:center;'>LLM Writing Assistant</h2>", unsafe_allow_html=True)

    cols = st.columns(2)

    # Texteingabe (links)
    with cols[0]:
        user_input = st.text_area(
            label="",
            placeholder="Type your text here and choose an action for the AI to perform.",
            height=320,
            label_visibility="collapsed"
        )

    # AI-Ausgabe (rechts)
    with cols[1]:
        if "ai_response" not in st.session_state:
            st.session_state.ai_response = ""

        st.text_area(
            label="",
            value=st.session_state.ai_response,
            placeholder="AI response will appear here...",
            height=320,
            label_visibility="collapsed",
            disabled=True
        )

    # Diff anzeigen
    if user_input.strip() and st.session_state.ai_response.strip():
        diff_result = display_diff(user_input, st.session_state.ai_response)
        st.markdown("### Unterschiede zum Originaltext")
        st.text_area("Diff-Ansicht", value=diff_result, height=200)

    # Aktionen
    actions = [
        ("Paraphrasing", "full", 0.7),
        ("Grammar Checker", "grammar", 0.5),
        ("Proofreading", "full", 0.5),
        ("Citation", "full", 0.8),
        ("Summarizer", "full", 0.6),
    ]

    # Symmetrische Button-Anordnung: [Rand, B1, B2, B3, B4, B5, Rand]
    button_cols = st.columns([0.1, 1, 1, 1, 1, 1, 0.1])
    for i, (label, mode, weight) in enumerate(actions):
        with button_cols[i + 1]:  # Buttons in Spalten 1 bis 5
            button_clicked = st.button(label, key=f"btn_{i}")
            if button_clicked:
                if not user_input.strip():
                    st.warning("Please enter some text before selecting an action.")
                else:
                    with st.spinner("Verarbeite deinen Text..."):
                        result = send_text_for_assistance(user_input, mode, weight)
                        st.session_state.ai_response = result.get("assisted_text", "")

# App starten
if __name__ == "__main__":
    main()
