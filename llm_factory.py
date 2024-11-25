from typing import Optional, Dict, Any
from anthropic import Anthropic_LLM, Anthropic_Chatbot
from cohere import Cohere_LLM, Cohere_Chatbot
from grok import Grok_LLM, Grok_Chatbot
from mistral import Mistral_LLM, Mistral_Chatbot
from openai import OpenAI_LLM, OpenAI_Chatbot

class LLMFactory:
    """Factory class to create LLM instances based on provider"""
    
    PROVIDERS = {
        "anthropic": (Anthropic_LLM, Anthropic_Chatbot),
        "cohere": (Cohere_LLM, Cohere_Chatbot),
        "grok": (Grok_LLM, Grok_Chatbot),
        "mistral": (Mistral_LLM, Mistral_Chatbot),
        "openai": (OpenAI_LLM, OpenAI_Chatbot)
    }

    @classmethod
    def create_llm(cls, provider: str, **kwargs) -> Any:
        """Create an LLM instance for the specified provider"""
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Available providers: {list(cls.PROVIDERS.keys())}")
        
        llm_class = cls.PROVIDERS[provider][0]
        return llm_class(**kwargs)

    @classmethod
    def create_chatbot(cls, provider: str, llm: Any, **kwargs) -> Any:
        """Create a Chatbot instance for the specified provider"""
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Available providers: {list(cls.PROVIDERS.keys())}")
        
        chatbot_class = cls.PROVIDERS[provider][1]
        return chatbot_class(llm=llm, **kwargs)

class MultiProviderChatbot:
    """Class to manage multiple chatbots from different providers"""
    
    def __init__(self):
        self.chatbots = {}
        self.factory = LLMFactory()

    def add_chatbot(self, 
                   provider: str, 
                   name: str,
                   system_prompt: str = "You are a helpful assistant.",
                   verbose: bool = True,
                   llm_params: Optional[Dict] = None,
                   chatbot_params: Optional[Dict] = None):
        """Add a new chatbot instance for a specific provider"""
        llm_params = llm_params or {}
        chatbot_params = chatbot_params or {}

        # Create LLM instance
        llm = self.factory.create_llm(provider, **llm_params)
        
        # Create Chatbot instance
        chatbot = self.factory.create_chatbot(
            provider,
            llm=llm,
            system_prompt=system_prompt,
            verbose=verbose,
            name=name,
            **chatbot_params
        )
        
        self.chatbots[name] = {
            "provider": provider,
            "instance": chatbot
        }

    def get_chatbot(self, name: str) -> Any:
        """Get a specific chatbot instance by name"""
        if name not in self.chatbots:
            raise ValueError(f"Chatbot '{name}' not found")
        return self.chatbots[name]["instance"]

    def list_chatbots(self) -> Dict[str, str]:
        """List all available chatbots and their providers"""
        return {name: info["provider"] for name, info in self.chatbots.items()}

    def remove_chatbot(self, name: str):
        """Remove a chatbot instance"""
        if name in self.chatbots:
            del self.chatbots[name]
