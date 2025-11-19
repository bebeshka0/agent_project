from typing import Dict, Optional

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag_agent import build_rag_chain
from router_agent import build_router_chain


def initialize_tutor_chain() -> RunnablePassthrough:
    """
    Initialize the base ML tutor chain without access to external documents.
    """
    llm: ChatOllama = ChatOllama(model="phi3:mini")

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
                    "other ML concepts."
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


def main() -> None:
    st.set_page_config(page_title="ML Learning Agent", layout="wide")
    
    st.title("Machine Learning Learning Agent")
    st.caption("Multi-agent system for learning various machine learning topics")
    
    # Initialize all agents in session state
    if (
        "tutor_chain" not in st.session_state
        or "rag_chain" not in st.session_state
        or "router_chain" not in st.session_state
    ):
        with st.spinner("Initializing multi-agent system..."):
            st.session_state.tutor_chain = initialize_tutor_chain()
            st.session_state.rag_chain = build_rag_chain()
            st.session_state.router_chain = build_router_chain()
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input field
    user_question: Optional[str] = st.chat_input("Ask about machine learning topics...")
    
    if user_question:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Decide which agent should handle the question
                    route: str = st.session_state.router_chain.invoke(user_question)

                    if route == "RAG":
                        rag_result: Dict[str, str] = st.session_state.rag_chain.invoke(
                            user_question
                        )
                        answer: str = rag_result.get("answer", "")
                        context: str = rag_result.get("context", "")

                        st.caption("Router: used RAG agent (documents-based answer)")
                        st.markdown(answer)

                        if context:
                            with st.expander("Retrieved context from documents"):
                                st.text(context)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                    else:
                        # Default to tutor agent
                        response: str = st.session_state.tutor_chain.invoke(
                            user_question
                        )
                        st.caption("Router: used TUTOR agent (theoretical explanation)")
                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                except Exception as e:
                    error_message: str = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()
