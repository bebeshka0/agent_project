import os
from typing import List, Optional

import streamlit as st
import requests

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

def main() -> None:
    st.set_page_config(page_title="ML Learning Agent", layout="wide")

    st.title("Machine Learning Learning Agent")
    st.caption("Multi-agent system for learning various machine learning topics")

    # Sidebar for file uploads
    with st.sidebar:
        st.header("Document Upload")
        st.markdown(
            "Upload your own PDF documents to add them to the knowledge base."
        )
        uploaded_files = st.file_uploader(
            "Choose PDF files", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files and st.button("Ingest Documents"):
            with st.spinner("Uploading and starting ingestion..."):
                try:
                    # Prepare files for upload
                    files_to_send = []
                    for uploaded_file in uploaded_files:
                        # tuple format: (filename, file_object, content_type)
                        files_to_send.append(
                            ("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"))
                        )

                    # Send POST request to FastAPI
                    response = requests.post(f"{API_URL}/ingest", files=files_to_send)
                    response.raise_for_status()
                    
                    data = response.json()
                    st.success(data.get("message", "Documents uploaded successfully!"))
                    st.info("Processing is running in the background. You can continue chatting.")

                except requests.exceptions.ConnectionError:
                     st.error("Error: Could not connect to the backend API.")
                except Exception as e:
                    st.error(f"Error uploading documents: {str(e)}")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_question: Optional[str] = st.chat_input("Ask about machine learning topics...")
    
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call FastAPI backend
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={"question": user_question}
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    answer: str = data.get("answer", "")
                    source: str = data.get("source", "TUTOR")
                    context: Optional[str] = data.get("context")

                    if source == "RAG":
                        st.caption("Router: used RAG agent (documents-based answer)")
                        st.markdown(answer)

                        if context:
                            with st.expander("Retrieved context from documents"):
                                st.text(context)
                    else:
                        st.caption("Router: used TUTOR agent (theoretical explanation)")
                        st.markdown(answer)
                        
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except requests.exceptions.ConnectionError:
                    error_message = "Error: Could not connect to the backend API. Is it running?"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()
