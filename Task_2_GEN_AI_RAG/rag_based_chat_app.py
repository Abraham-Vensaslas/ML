"""
This script provides a Retrieval-Augmented Generation (RAG)-based chatbot using the open source light weight 'gemini-2.0-flash-lite' model and FAISS vector store. The chatbot performs the following operations:

1. **Embedding and Vector Store**: 
   - Loads precomputed document embeddings from a FAISS index.
   - Uses `HuggingFaceEmbeddings` with a model (`nomic-ai/nomic-embed-text-v1`) to convert documents into vector embeddings.
   
2. **Gemini Model**:
   - Loads the Gemini model (`gemini-2.0-flash-lite`) for generating natural language responses.
   - The model is loaded once and used to generate answers based on retrieved context from the FAISS vector store.

3. **RAG-based Query Answering**:
   - A function `ask_llama_rag(query, chat_history)` that takes a user query, searches for relevant documents in the vector store, and generates an answer using TinyLLaMA.
   
4. **Interactive Chat Loop**:
   - An interactive chat loop allows the user to input questions, receive answers, and continue the conversation with context.
   - The conversation history is appended for continuity, and the assistant answers based on the context of the query and previous chat history.
   
The model supports a variety of document types, such as PDF, DOCX, XLSX, and CSV, stored in a FAISS vector store for efficient retrieval. 
"""


from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import os
import google.generativeai as genai
# from dotenv import load_dotenv
# # Load environment variables from .env file
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = "AIzaSyBjEvwM6ikap4zfX-kvh5hYC4uRv-bF3P4"
genai.configure(api_key="AIzaSyBjEvwM6ikap4zfX-kvh5hYC4uRv-bF3P4")

# ----------------- Load FAISS once ------------------
# ----------------- ✅ CACHED LOADING FUNCTIONS ------------------ #
@st.cache_resource(show_spinner="🔄 Loading vector store...")
def load_vectorstore():
    model_name = "nomic-ai/nomic-embed-text-v1"
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': False}
    )

    return FAISS.load_local(
        "faiss_nomic_index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel('gemini-2.0-flash-lite')


def ask_gemini_rag(question, chat_history):

   # Gemini model setup (once, globally)
   model = genai.GenerativeModel('gemini-2.0-flash-lite')

   vectorstore = load_vectorstore()
   model = get_gemini_model()

   # 1. Retrieve relevant chunks from vector store
   docs = vectorstore.similarity_search(question, k=4)  # Get top 4 matches
   context = "\n\n".join([doc.page_content for doc in docs])
   # 2. Construct a prompt
   full_chat = "\n".join(chat_history)
   prompt = f"""
            You are an advanced assistant, committed to providing responses strictly based on the context below. Your answers should always be grounded in the provided information, avoiding assumptions, guesses, or fabrications.

            Instructions:
            1. **Thoroughly examine the provided context** and extract the most relevant details to form a comprehensive answer.
            2. Always try to give a good explanation while answering.
            3. **Do not infer or guess** any information not explicitly mentioned in the context. If the answer is unclear or incomplete, acknowledge this directly and indicate the limitations.
            4. If the context is insufficient to fully answer the question, offer a brief, honest acknowledgment of the missing information. Do **not speculate**.
            5. If the context contains **multiple possible answers**, synthesize a coherent response, mentioning all relevant points. Provide **adequate detail** for a thoughtful, meaningful answer.
            6. Always aim to structure your response in a **clear, coherent**, and logically-organized manner, ensuring all critical context is highlighted and integrated into the answer.
            7. If the question falls outside the scope of the context (e.g., the documents are about banking but the question is about football), return **the keyword 'False'** so that a web search can be activated automatically. Do not attempt to answer such questions without related context.

            Context:
            {context}

            Conversation so far:
            {full_chat}


            Question: {question}

            Answer (Be detailed, concise, and clear, **only based on the provided context**. Avoid overly brief responses and never guess):

            

            """


   # 3. Ask Gemini for a response
   response = model.generate_content(prompt)    
   return response.text


# ----------------- Chat Loop ------------------
import streamlit as st
st.set_page_config(
    page_title="Gem Fin Assistant",
    page_icon="🤖",  # You can use an emoji or link to a .ico/.png
)

chat_history = []

st.title("💬 Chat with your Gemini Finance Assistant")

# Input box
user_input = st.text_input("You:")

if user_input:
    if user_input.lower() in ["exit", "quit"]:
        st.write("👋 Ending chat.")
    else:
        # Show spinner while processing
        with st.spinner("🔍 Searching the documents..."):
            answer = ask_gemini_rag(user_input, chat_history)

        # Show result
        st.write("Assistant:", answer)

        # Append to local history
        chat_history.append(f"<|user|>\n{user_input}")
        chat_history.append(f"<|assistant|>\n{answer}")