import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def build_tutor_chain() -> RunnablePassthrough:
    """
    Constructs the Tutor Chain for general ML explanations.
    """
    llm = ChatOpenAI(
        model=os.getenv("XAI_MODEL", "grok-4-1-fast-non-reasoning"),
        base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        api_key=os.getenv("XAI_API_KEY"),
    )

    prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert AI tutor specializing in machine learning education. "
                    "Your role is to teach various machine learning topics clearly and "
                    "comprehensively. Provide detailed explanations, examples, and practical "
                    "insights. Adapt your teaching style to the user's level of understanding. "
                    "Cover topics such as supervised learning, unsupervised learning, deep "
                    "learning, neural networks, model evaluation, feature engineering, and "
                    "other ML concepts. "
                    "Do not use emojis, tables, or complex formatting unless explicitly "
                    "requested by the user."
                ),
            ),
            ("human", "{user_input}"),
        ]
    )

    chain: RunnablePassthrough = (
        {"user_input": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain

