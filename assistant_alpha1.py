import json
from typing import Optional, List, Dict, Union, Any
from datetime import datetime
import os
import re

class GlobalAIAssistant:
    def __init__(
        self,
        llm_provider: str = "openai",  # openai, anthropic, cohere, or mistral
        embedding_provider: str = "openai",  # openai, cohere, mistral, or voyage
        llm_config: Dict[str, Any] = None,
        embedding_config: Dict[str, Any] = None,
        system_prompt: str = "You are a helpful assistant.",
        context_window: int = 5,
        similar_chunks: int = 3,
        verbose: bool = True,
        show_context: bool = False
    ):
        self.llm_provider = llm_provider.lower()
        self.embedding_provider = embedding_provider.lower()
        self.context_window = context_window
        self.similar_chunks = similar_chunks
        self.verbose = verbose
        self.show_context = show_context
        
        # Initialize LLM
        self.llm = self._initialize_llm(llm_config or {})
        
        # Initialize Embedding model
        self.embedding_model = self._initialize_embedding(embedding_config or {})
        
        # Initialize chatbot
        self.chatbot = self._initialize_chatbot(system_prompt)
        
        # Context markers for identifying retrieval context in messages
        self.context_start_marker = "### Relevant Context ###\n"
        self.context_end_marker = "\n### End Context ###\n"
        self.question_marker = "\nQuestion: "

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

    def _get_conversation_history(self) -> List[Dict]:
        """Get recent conversation history without retrieval context"""
        history = []
        for msg in self.chatbot.history[-self.context_window:]:
            if msg.get("role") in ["user", "assistant"]:
                content = msg.get("content", "")
                
                # Handle different message formats
                if isinstance(content, list):
                    # Handle OpenAI format
                    content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
                elif isinstance(content, dict):
                    # Handle Cohere format
                    content = content.get("text", "")
                
                # Remove context from user messages
                if msg["role"] == "user":
                    content = self._remove_context_from_message(content)
                
                history.append({"role": msg["role"], "content": content})
        return history

    def _remove_context_from_message(self, message: str) -> str:
        """Remove context section from message"""
        if self.context_start_marker in message and self.context_end_marker in message:
            pattern = f"{self.context_start_marker}.*?{self.context_end_marker}"
            message = re.sub(pattern, "", message, flags=re.DOTALL)
            # Remove Question marker if present
            message = message.replace(self.question_marker, "")
        return message.strip()

    def _get_similar_chunks(self, query: str, index_name: str) -> List[str]:
        """Get similar chunks from the embedding index"""
        try:
            results = self.embedding_model.search(index_name, query, k=self.similar_chunks)
            return [result["text"] for result in results]
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not retrieve similar chunks: {e}")
            return []

    def _format_message_with_context(self, message: str, chunks: List[str]) -> str:
        """Format message with retrieval context"""
        if not chunks:
            return message
            
        context = "\n".join(f"- {chunk}" for chunk in chunks)
        return (
            f"{self.context_start_marker}"
            f"{context}"
            f"{self.context_end_marker}"
            f"{self.question_marker}{message}"
        )
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
        """Process user message with context management"""
        original_message = message  # Keep a copy of the original message
        
        # Get similar chunks if index_name is provided
        if index_name:
            # Get clean conversation history
            history = self._get_conversation_history()
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            
            # Combine history and current message for context search
            query = f"{history_text}\n{message}" if history_text else message
            
            # Get new retrieval context
            similar_chunks = self._get_similar_chunks(query, index_name)
            
            # Format message with new context
            enhanced_message = self._format_message_with_context(message, similar_chunks)
        else:
            enhanced_message = message
        
        # Get response from chatbot
        response = self.chatbot(enhanced_message)
        
        # If show_context is False, replace the enhanced message with the original message in history
        if not self.show_context:
            for msg in self.chatbot.history:
                if msg["role"] == "user" and self.context_start_marker in msg.get("content", ""):
                    if isinstance(msg["content"], str):
                        msg["content"] = original_message
                    elif isinstance(msg["content"], list):  # For OpenAI format
                        msg["content"] = [{"type": "text", "text": original_message}]
                    elif isinstance(msg["content"], dict):  # For Cohere format
                        msg["content"]["text"] = original_message
        
        return response

    def start_new_conversation(self):
        """Start a new conversation"""
        self.chatbot.start_new_conversation()

    def load_conversation(self, conversation_id: str):
        """Load a specific conversation"""
        self.chatbot.load_conversation(conversation_id)

    def list_conversations(self) -> List[str]:
        """List all conversations"""
        return self.chatbot.list_conversations()
