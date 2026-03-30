from typing import Dict, List

import openai
import streamlit as st

from utils.gemini import convert_messages_format
from settings import (
    USE_GEMINI,
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    GPT_BASE,
    GPT_VERSION,
    GPT_KEY,
    GPT_ENGINE,
)
from classes.corner_description import CornerDescription


class CornerChat:
    def __init__(self, state_key: str, description: CornerDescription):
        self.state_key = str(state_key)
        self.transcript_key = f"{self.state_key}::last_transcript"
        self.description = description

        if self.state_key not in st.session_state:
            st.session_state[self.state_key] = []
        if self.transcript_key not in st.session_state:
            st.session_state[self.transcript_key] = []

    @property
    def history(self) -> List[Dict[str, str]]:
        return st.session_state[self.state_key]

    @property
    def last_transcript(self) -> List[Dict[str, str]]:
        return st.session_state[self.transcript_key]

    def ask(self, user_prompt: str) -> str:
        self.history.append({"role": "user", "content": str(user_prompt)})
        llm_messages = self.description.base_messages() + self.history
        st.session_state[self.transcript_key] = llm_messages
        answer = self._generate(llm_messages)
        self.history.append({"role": "assistant", "content": answer})
        return answer

    @staticmethod
    def _generate(messages: List[Dict[str, str]]) -> str:
        use_gemini = bool(USE_GEMINI and GEMINI_API_KEY and GEMINI_CHAT_MODEL)
        if use_gemini:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(messages)
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"],
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])
            return response.text

        missing = [
            key
            for key, value in {
                "GPT_BASE": GPT_BASE,
                "GPT_VERSION": GPT_VERSION,
                "GPT_KEY": GPT_KEY,
                "GPT_ENGINE/GPT_CHAT_MODEL": GPT_ENGINE,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "OpenAI/Azure config is incomplete. Missing: "
                + ", ".join(missing)
                + ". If you want Gemini, set USE_GEMINI=true and provide GEMINI_API_KEY."
            )

        openai.api_type = "azure"
        openai.api_base = GPT_BASE
        openai.api_version = GPT_VERSION
        openai.api_key = GPT_KEY
        response = openai.ChatCompletion.create(engine=GPT_ENGINE, messages=messages)
        return str(response["choices"][0]["message"]["content"])
