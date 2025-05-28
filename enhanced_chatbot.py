import os
import shutil
from typing import List, Optional, Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store and embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LLM and Chat logic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage

# --- CONFIGURATION ---
class ChatbotConfig:
    """Configuration settings for the chatbot."""
    DOCUMENT_SOURCE_PATH: str = "./documents"
    VECTOR_STORE_PATH: str = "./vector_store"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_PROVIDER: str = "Groq"
    MODEL_OLLAMA: str = "llama3"
    MODEL_GROQ: str = "llama3-8b-8192"
    MODEL_OPENAI: str = "gpt-4o"

class Chatbot:
    """An enhanced, conversational RAG chatbot with robust document loading."""

    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.vector_store: Optional[FAISS] = None
        self.llm: Optional[Any] = None
        self.rag_chain = None
        self.chat_history: List[HumanMessage | AIMessage] = []

        self._initialize_llm()
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL_NAME)
        self._load_or_create_vector_store()
        self._create_rag_chain()

    def _initialize_llm(self):
        provider = self.config.LLM_PROVIDER.lower()
        print(f"Initializing LLM provider: {self.config.LLM_PROVIDER}")
        if provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            self.llm = ChatOllama(model=self.config.MODEL_OLLAMA, temperature=0.3)
        elif provider == "groq":
            from langchain_groq import ChatGroq
            os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "") # Ensure you set this key
            self.llm = ChatGroq(model=self.config.MODEL_GROQ, temperature=0.3, api_key="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")
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
            self.llm = ChatOpenAI(model=self.config.MODEL_OPENAI, temperature=0.3)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.LLM_PROVIDER}")

    def _load_documents(self) -> List:
        """
        Loads documents from the source directory using specific loaders for each file type.
        """
        print("Loading documents using specialized loaders...")
        all_docs = []
        source_dir = self.config.DOCUMENT_SOURCE_PATH
        
        for file_name in os.listdir(source_dir):
            file_path = os.path.join(source_dir, file_name)
            loader = None
            try:
                if file_name.lower().endswith(".pdf"):
                    print(f"-> Loading PDF: {file_name}")
                    loader = PyPDFLoader(file_path)
                elif file_name.lower().endswith(".docx"):
                    print(f"-> Loading DOCX: {file_name}")
                    loader = Docx2txtLoader(file_path)
                elif file_name.lower().endswith(".txt"):
                    print(f"-> Loading TXT: {file_name}")
                    loader = TextLoader(file_path)

                if loader:
                    all_docs.extend(loader.load())
            except Exception as e:
                print(f"!! FAILED to load {file_name}. Error: {e}. Skipping file.")

        print(f"Successfully loaded {len(all_docs)} pages/documents.")
        return all_docs
        
    def _create_vector_store(self):
        print("No existing vector store found. Creating a new one...")
        documents = self._load_documents()
        if not documents:
            raise ValueError("No documents were loaded successfully. Cannot create vector store.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Creating vector store from {len(chunks)} chunks...")
        self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
        print(f"Saving vector store to '{self.config.VECTOR_STORE_PATH}'...")
        self.vector_store.save_local(self.config.VECTOR_STORE_PATH)

    def _load_or_create_vector_store(self):
        if os.path.exists(self.config.VECTOR_STORE_PATH):
            print(f"Loading existing vector store from '{self.config.VECTOR_STORE_PATH}'...")
            self.vector_store = FAISS.load_local(
                self.config.VECTOR_STORE_PATH, 
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            self._create_vector_store()

    def _create_rag_chain(self):
        print("Creating RAG chain...")
        retriever = self.vector_store.as_retriever()
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)
        qa_system_prompt = ("You are an expert assistant for question-answering tasks. Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be concise and helpful.\n\nContext:\n{context}")
        qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
        Youtube_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)
        print("Chatbot is ready!")

    def start_chat(self):
        print("\n--- AI Document Chatbot ---")
        print("Ask questions about your documents. Type 'exit' to quit or 'reset' to clear history.")
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                if user_input.lower() == 'reset':
                    self.chat_history = []
                    print("Chat history cleared.")
                    continue
                if not user_input:
                    continue
                print("Bot: ", end="", flush=True)
                response_stream = self.rag_chain.stream({"input": user_input, "chat_history": self.chat_history})
                full_response = ""
                source_documents = []
                for chunk in response_stream:
                    if "answer" in chunk:
                        print(chunk["answer"], end="", flush=True)
                        full_response += chunk["answer"]
                    if "context" in chunk:
                        source_documents.extend(chunk["context"])
                print() 
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=full_response))
                if source_documents:
                    unique_sources = {doc.metadata.get('source', 'Unknown') for doc in source_documents}
                    print("\nSources Used:")
                    for source in sorted(list(unique_sources)):
                        print(f"- {os.path.basename(source)}")
            except KeyboardInterrupt:
                print("\nInterrupted. Exiting...")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    config = ChatbotConfig()
    chatbot = Chatbot(config)
    chatbot.start_chat()