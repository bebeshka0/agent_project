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
    """Route a user query to the most relevant datasource."""
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
                    "Use 'TUTOR' if the question is about general machine learning "
                    "concepts, definitions, or explanations.\n"
                    "Use 'RAG' if the question refers to specific uploaded "
                    "documents, articles, PDFs, results, formulas, or content that "
                    "likely exists in those documents.\n"
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

    # Chain: Prompt -> LLM -> Pydantic Parser
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


