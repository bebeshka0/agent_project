from typing import Dict, Optional
import string

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama


ROUTER_TUTOR: str = "TUTOR"
ROUTER_RAG: str = "RAG"
VALID_ROUTES = {ROUTER_TUTOR, ROUTER_RAG}


def build_router_chain() -> Runnable:
    llm: ChatOllama = ChatOllama(model="phi3:mini")

    prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a router that classifies user questions into one of "
                    "two categories: TUTOR or RAG "
                    "Return exactly one word: 'TUTOR' if the question is about "
                    "general machine learning concepts, definitions, or explanations. "
                    "Return 'RAG' if the question refers to specific uploaded "
                    "documents, articles, PDFs, results, formulas, or content that "
                    "likely exists in those documents."
                ),
            ),
            (
                "human",
                "Question:\n{question}\n\nAnswer with a single word: TUTOR or RAG.",
            ),
        ]
    )

    def _route_call(question: str) -> str:
        messages = prompt_template.format_messages(question=question)
        response = llm.invoke(messages)
        raw_text: str = StrOutputParser().invoke(response).strip()

        # Use only the first token (cleaned from punctuation) to make the router robust
        # to verbose or slightly malformed outputs.
        if raw_text:
            first_word: str = raw_text.split()[0].strip(string.punctuation)
            first_token: str = first_word.upper()
        else:
            first_token = ""

        if first_token not in VALID_ROUTES:
            # Fallback to TUTOR for safety
            route: str = ROUTER_TUTOR
        else:
            route = first_token

        print(f"Router decision: {route} (raw: {raw_text})")
        return route

    class RouterRunnable(Runnable):

        def invoke(self, input: str, config: Optional[Dict] = None) -> str:
            return _route_call(question=input)

    return RouterRunnable()


