import json
from typing import Optional, List, Dict, Union, Any
from datetime import datetime
import os

class GlobalAIAssistant:
    def __init__(
        self,
        llm_provider: str = "openai",  # openai, anthropic, cohere, or mistral
        embedding_provider: str = "openai",  # openai, cohere, mistral, or voyage
        llm_config: Dict[str, Any] = None,
        embedding_config: Dict[str, Any] = None,
        base_system_prompt: str = "You are a helpful assistant.",
        context_window: int = 5,  # Number of previous messages to include
        similar_chunks: int = 3,  # Number of similar chunks to retrieve
        verbose: bool = True
    ):
        self.llm_provider = llm_provider.lower()
        self.embedding_provider = embedding_provider.lower()
        self.context_window = context_window
        self.similar_chunks = similar_chunks
        self.verbose = verbose
        self.base_system_prompt = base_system_prompt
        
        # Initialize LLM
        self.llm = self._initialize_llm(llm_config or {})
        
        # Initialize Embedding model
        self.embedding_model = self._initialize_embedding(embedding_config or {})
        
        # Initialize chatbot with base system prompt
        self.chatbot = self._initialize_chatbot(self.base_system_prompt)

    def _initialize_llm(self, config: Dict[str, Any]):
        """Initialize the LLM based on provider"""
        if self.llm_provider == "openai":
            from openai import OpenAI_LLM
            return OpenAI_LLM(**config)
        elif self.llm_provider == "anthropic":
            from anthropic import Anthropic_LLM
            return Anthropic_LLM(**config)
        elif self.llm_provider == "cohere":
            from cohere import Cohere_LLM
            return Cohere_LLM(**config)
        elif self.llm_provider == "mistral":
            from mistral import Mistral_LLM
            return Mistral_LLM(**config)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _initialize_embedding(self, config: Dict[str, Any]):
        """Initialize the embedding model based on provider"""
        if self.embedding_provider == "openai":
            from openai_embedding import OpenAI_Embedding
            return OpenAI_Embedding(**config)
        elif self.embedding_provider == "cohere":
            from cohere_embedding import Cohere_Embedding
            return Cohere_Embedding(**config)
        elif self.embedding_provider == "mistral":
            from mistral_embedding import Mistral_Embedding
            return Mistral_Embedding(**config)
        elif self.embedding_provider == "voyage":
            from voyage_embedding import Voyage_Embedding
            return Voyage_Embedding(**config)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

    def _initialize_chatbot(self, system_prompt: str):
        """Initialize the chatbot based on LLM provider"""
        if self.llm_provider == "openai":
            from openai import OpenAI_Chatbot
            return OpenAI_Chatbot(self.llm, system_prompt=system_prompt, verbose=self.verbose)
        elif self.llm_provider == "anthropic":
            from anthropic import Anthropic_Chatbot
            return Anthropic_Chatbot(self.llm, system_prompt=system_prompt, verbose=self.verbose)
        elif self.llm_provider == "cohere":
            from cohere import Cohere_Chatbot
            return Cohere_Chatbot(self.llm, system_prompt=system_prompt, verbose=self.verbose)
        elif self.llm_provider == "mistral":
            from mistral import Mistral_Chatbot
            return Mistral_Chatbot(self.llm, system_prompt=system_prompt, verbose=self.verbose)

    def _get_similar_chunks(self, query: str, index_name: str) -> List[str]:
        """Get similar chunks from the embedding index"""
        try:
            results = self.embedding_model.search(index_name, query, k=self.similar_chunks)
            return [result["text"] for result in results]
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not retrieve similar chunks: {e}")
            return []

    def _update_system_prompt(self, chunks: List[str]):
        """Update system prompt with new context"""
        if not chunks:
            new_system_prompt = self.base_system_prompt
        else:
            context = "\n".join([f"- {chunk}" for chunk in chunks])
            new_system_prompt = f"""{self.base_system_prompt}

Here is the relevant information you should use to answer the user's questions:

{context}

Please use this information to provide accurate answers. If the information provided doesn't fully address the user's question, you can supplement it with your general knowledge while prioritizing the provided context."""
        
        # Start new conversation with updated system prompt
        self.chatbot.start_new_conversation()
        self.chatbot.system_prompt = new_system_prompt
        
        # Update first message in history with new system prompt
        if self.chatbot.history and self.chatbot.history[0]["role"] == "system":
            self.chatbot.history[0]["content"] = new_system_prompt

    def create_knowledge_base(self, texts: List[str], index_name: str):
        """Create a knowledge base from texts"""
        try:
            self.embedding_model.create_faiss_index(index_name, texts)
            if self.verbose:
                print(f"Successfully created knowledge base: {index_name}")
        except Exception as e:
            raise Exception(f"Failed to create knowledge base: {e}")

    def update_knowledge_base(self, new_texts: List[str], index_name: str):
        """Update existing knowledge base with new texts"""
        try:
            self.embedding_model.update_index(index_name, new_texts)
            if self.verbose:
                print(f"Successfully updated knowledge base: {index_name}")
        except Exception as e:
            raise Exception(f"Failed to update knowledge base: {e}")

    def chat(self, message: str, index_name: Optional[str] = None) -> str:
        """Process user message with dynamic system prompt updates"""
        # Get similar chunks and update system prompt if index_name is provided
        if index_name:
            similar_chunks = self._get_similar_chunks(message, index_name)
            self._update_system_prompt(similar_chunks)
        
        # Get response from chatbot
        response = self.chatbot(message)
        return response

    def start_new_conversation(self):
        """Start a new conversation with base system prompt"""
        self._update_system_prompt([])  # Reset to base system prompt

    def load_conversation(self, conversation_id: str):
        """Load a specific conversation"""
        self.chatbot.load_conversation(conversation_id)

    def list_conversations(self) -> List[str]:
        """List all conversations"""
        return self.chatbot.list_conversations()
