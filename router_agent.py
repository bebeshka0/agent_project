import os
from typing import Dict, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


ROUTER_TUTOR: str = "TUTOR"
ROUTER_RAG: str = "RAG"


class RouteDecision(BaseModel):
    datasource: Literal["TUTOR", "RAG"] = Field(
        ...,
        description=(
            "Given a user question choose to route it to 'TUTOR' if it is about general "
            "concepts, or 'RAG' if it is about specific documents."
        )
    )


def build_router_chain() -> Runnable:
    llm = ChatOpenAI(
        model=os.getenv("XAI_MODEL", "grok-4-1-fast-non-reasoning"),
        base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        api_key=os.getenv("XAI_API_KEY"),
    )

    parser = PydanticOutputParser(pydantic_object=RouteDecision)

    prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a router that classifies user questions into one of "
                    "two categories: TUTOR or RAG.\n"
                    
                    "RULES:\n"
                    "1. Use 'RAG' (Retrieval-Augmented Generation) if the user:\n"
                    "   Asks about specific documents, PDF, files, articles, or books.\n"
                    "   Asks to find something 'in the text', 'in the book', 'according to the article'.\n"
                    "   Asks about specific details that imply searching a knowledge base.\n"
                    
                    "2. Use 'TUTOR' if the question is:\n"
                    "   A general concept explanation (e.g., 'What is a neural network?').\n"
                    "   A request for code or examples NOT tied to a specific document.\n"
                    "   Casual conversation or greetings.\n"

                    "\n"
                    "You must answer strictly using the specified format instructions.\n"
                    "{format_instructions}"
                ),
            ),
            (
                "human",
                "Question:\n{question}",
            ),
        ]
    )

    chain = prompt_template | llm | parser

    def _route_call(question: str) -> str:
        try:
            # Pass format_instructions to the prompt
            decision: RouteDecision = chain.invoke({
                "question": question,
                "format_instructions": parser.get_format_instructions()
            })
            route = decision.datasource
        except Exception as e:
            print(f"Router parsing error: {e}. Fallback to TUTOR.")
            route = ROUTER_TUTOR
        
        print(f"Router decision: {route}")
        return route

    class RouterRunnable(Runnable):

        def invoke(self, input: str, config: Optional[Dict] = None) -> str:
            return _route_call(question=input)

    return RouterRunnable()


