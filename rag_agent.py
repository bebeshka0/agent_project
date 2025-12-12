import os
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from flashrank import Ranker, RerankRequest
from rag_vector_store import get_retriever


def _format_context(documents: List[Document]) -> str:
 
    formatted_chunks: List[str] = []

    for index, document in enumerate(documents, start=1):
        source: str = str(document.metadata.get("original_filename") or document.metadata.get("source", "unknown_source"))
        page: str = str(document.metadata.get("page", document.metadata.get("page_number", "unknown_page")))
        doc_id: str = str(document.metadata.get("doc_id", "unknown_doc_id"))
        version: str = str(document.metadata.get("version", "unknown_version"))
        preview: str = document.page_content

        chunk_header: str = f"[Chunk {index}] Doc: {doc_id} (v{version}), Source: {source}, Page: {page}"
        formatted_chunks.append(f"{chunk_header}\n{preview}")

    return "\n\n".join(formatted_chunks)

def rerank_documents(question: str, documents: List[Document], top_n: int = 5) -> List[Document]:
    if not documents:
        return []
    
    print(f"Reranking {len(documents)} documents for question: {question}")

    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
    
    passages = [
        {"id": str(i), "text": doc.page_content, "metadata": doc.metadata} for i, doc in enumerate(documents)
    ]

    rerank_request = RerankRequest(query = question, passages = passages)
    results = ranker.rerank(rerank_request)

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]

    reranked_docs = []
    for result in results:
        doc = Document(page_content=result["text"], metadata=result["metadata"])
        doc.metadata["rerank_score"] = result["score"]
        reranked_docs.append(doc)

    return reranked_docs

def retrieve_rag_context(question: str, k: int = 8) -> Tuple[str, List[Document]]:

    initial_k = k*4

    retriever = get_retriever(k=initial_k)

    if retriever is None:
        print("RAG retriever is not available. Check database connection and configuration.")
        return "", []
    
    if hasattr(retriever, "get_relevant_documents"):
        documents: List[Document] = retriever.get_relevant_documents(question)
    else:
        documents = retriever.invoke(question)
    
    print(f"Retrieved {len(documents)} documents from PGVector.")

    reranked_docs = rerank_documents(question, documents, top_n = k)

    print("Top Reranked documents:")
    for index, doc in enumerate(reranked_docs, start=1):
        source: str = str(doc.metadata.get("source", "unknown_source"))
        score: float = doc.metadata.get("rerank_score", "N/A")
        print(f"[Chunk {index}] Rank {index}], Source: {source}, Score: {score}")
    
    context: str = _format_context(reranked_docs)
    return context, reranked_docs


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


