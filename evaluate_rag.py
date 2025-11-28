import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
# Updated imports for newer Ragas versions
from ragas.metrics import (
    faithfulness,
    answer_relevancy,  # <-- Note the 'y'
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings # Use local embeddings

# Import our RAG chain
from rag_agent import build_rag_chain, retrieve_rag_context

def load_test_dataset(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("Loading test dataset...")
    test_data = load_test_dataset("test_dataset.json")
    
    # Initialize RAG chain
    rag_chain = build_rag_chain()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    print(f"Running RAG on {len(test_data)} test questions...")
    
    for item in test_data:
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"Processing: {question}")
        
        # 1. Get answer from our RAG
        result = rag_chain.invoke(question)
        answer = result["answer"]
        
        # 2. Re-retrieve documents specifically to get them as list of strings for Ragas
        _, docs = retrieve_rag_context(question, k=5) 
        context_list = [doc.page_content for doc in docs]
        
        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(ground_truth)

    # Prepare dataset for Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data_dict)
    
    print("Starting evaluation with Ragas metrics...")
    
    # Initialize LLM (Judge)
    eval_llm = ChatOpenAI(
        model=os.getenv("XAI_MODEL", "grok-4-1-fast-non-reasoning"),
        base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        api_key=os.getenv("XAI_API_KEY"),
    )
    
    # Initialize Embeddings (Local/HuggingFace) - Critical for AnswerRelevance
    # Must match what you used for ingestion ideally, or just be a good generic model
    eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # No need to instantiate classes manually if importing instances
    
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=eval_llm, 
        embeddings=eval_embeddings
    )
    
    print("\nEvaluation Results:")
    print(results)
    
    # Save details to CSV
    df = results.to_pandas()
    df.to_csv("rag_evaluation_results_after_reranking.csv", index=False)
    print("Detailed results saved to 'rag_evaluation_results_after_reranking.csv'")

if __name__ == "__main__":
    main()
