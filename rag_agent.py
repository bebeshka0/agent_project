import os
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from rag_vector_store import get_retriever


def _format_context(documents: List[Document]) -> str:
 
    formatted_chunks: List[str] = []

    for index, document in enumerate(documents, start=1):
        source: str = str(document.metadata.get("source", "unknown_source"))
        page: str = str(document.metadata.get("page", document.metadata.get("page_number", "unknown_page")))
        preview: str = document.page_content[:400].replace("\n", " ")

        chunk_header: str = f"[Chunk {index}] Source: {source}, Page: {page}"
        formatted_chunks.append(f"{chunk_header}\n{preview}")

    return "\n\n".join(formatted_chunks)


def retrieve_rag_context(question: str, k: int = 5) -> Tuple[str, List[Document]]:

    retriever = get_retriever(k=k)

    if retriever is None:
        print("RAG retriever is not available. Check database connection and configuration.")
        return "", []

    if hasattr(retriever, "get_relevant_documents"):
        documents: List[Document] = retriever.get_relevant_documents(question)
    else:
        documents = retriever.invoke(question)

    print("Retrieved documents from PGVector:")
    if not documents:
        print("No documents retrieved for this question.")
    else:
        for index, document in enumerate(documents, start=1):
            source: str = str(document.metadata.get("source", "unknown_source"))
            page: str = str(document.metadata.get("page", document.metadata.get("page_number", "unknown_page")))
            preview: str = document.page_content[:200].replace("\n", " ")
            print(f"[Chunk {index}] Source: {source}, Page: {page}")
            print(f"Preview: {preview}")
            print("-" * 80)

    context: str = _format_context(documents)
    return context, documents


def build_rag_chain() -> Runnable:
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
                    "You are a research assistant that answers strictly based on the "
                    "provided context from uploaded documents. If the answer is not "
                    "contained in the context, explicitly say that you do not know. "
                    "Do not use any external knowledge beyond the given context. "
                    "Do not use emojis, tables, or complex formatting unless explicitly "
                    "requested by the user."
                ),
            ),
            (
                "human",
                (
                    "Context:\n{context}\n\n"
                    "Question:\n{question}\n\n"
                    "Answer based only on the context above."
                ),
            ),
        ]
    )

    def _rag_call(question: str) -> Dict[str, str]:

        context, _ = retrieve_rag_context(question=question, k=5)
        messages = prompt_template.format_messages(context=context, question=question)
        response = llm.invoke(messages)
        parsed_response: str = StrOutputParser().invoke(response)
        return {"answer": parsed_response, "context": context}

    class RAGRunnable(Runnable):
        
        def invoke(self, input: str, config: Optional[Dict] = None) -> Dict[str, str]:
            return _rag_call(question=input)

    return RAGRunnable()


