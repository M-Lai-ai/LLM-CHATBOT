import json
import requests
import uuid
import os
from typing import Optional, Dict, List
from datetime import datetime
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

class Anthropic_LLM:
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0,
        max_tokens: int = 1024,
        stream: bool = True,
        top_p: float = 0,
        stop_sequences: Optional[List[str]] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.top_p = top_p
        self.stop_sequences = stop_sequences or []
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    def _make_request(self, messages: List[Dict], system: str) -> requests.Response:
        """Make request to Anthropic API"""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "system": system,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
            "stop_sequences": self.stop_sequences
        }

        return requests.post(url, headers=headers, json=payload, stream=self.stream)

class Anthropic_Chatbot:
    _chatbot_counter = 0  # Class variable to track number of chatbot instances

    def __init__(
        self,
        llm: Anthropic_LLM,
        system_prompt: str = "You are a helpful assistant.",
        verbose: bool = True,
        name: Optional[str] = None
    ):
        Anthropic_Chatbot._chatbot_counter += 1
        self.llm = llm
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.chatbot_id = Anthropic_Chatbot._chatbot_counter
        self.name = name or f"anthropic_chatbot_{self.chatbot_id}"
        self.conversation_folder = self._create_conversation_folder()
        self.history: List[Dict] = []
        self._initialize_conversation()

    def _create_conversation_folder(self) -> str:
        """Create and return the path to this chatbot's conversation folder"""
        base_folder = "conversations"
        chatbot_folder = f"{base_folder}/anthropic/{self.name}"
        os.makedirs(chatbot_folder, exist_ok=True)
        
        # Create or update chatbot metadata
        metadata = {
            "chatbot_id": self.chatbot_id,
            "name": self.name,
            "created_at": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "model": self.llm.model,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "stop_sequences": self.llm.stop_sequences
        }
        
        with open(f"{chatbot_folder}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return chatbot_folder

    def _initialize_conversation(self):
        """Initialize conversation history"""
        self.conversation_id = str(uuid.uuid4())
        self._save_conversation()

    def _save_conversation(self):
        """Save conversation history to JSON file"""
        filename = f"{self.conversation_folder}/conversation_{self.conversation_id}.json"
        
        conversation_data = {
            "conversation_id": self.conversation_id,
            "chatbot_name": self.name,
            "chatbot_id": self.chatbot_id,
            "timestamp": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "history": self.history
        }
        
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)

    def start_new_conversation(self):
        """Start a new conversation while maintaining chatbot identity"""
        self.conversation_id = str(uuid.uuid4())
        self.history = []
        self._save_conversation()
        if self.verbose:
            print(f"\nStarted new conversation with ID: {self.conversation_id}")

    def list_conversations(self) -> List[str]:
        """List all conversations for this chatbot"""
        conversations = [f for f in os.listdir(self.conversation_folder) 
                        if f.startswith('conversation_') and f.endswith('.json')]
        return conversations

    def load_conversation(self, conversation_id: str):
        """Load a specific conversation"""
        filename = f"{self.conversation_folder}/conversation_{conversation_id}.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.conversation_id = data["conversation_id"]
                self.history = data["history"]
                if self.verbose:
                    print(f"\nLoaded conversation: {conversation_id}")
        else:
            raise FileNotFoundError(f"Conversation {conversation_id} not found")

    def _prepare_messages(self, message: str) -> List[Dict]:
        """Prepare messages for API request"""
        self.history.append({
            "role": "user",
            "content": message
        })
        return self.history

    def _print_streaming_response(self, content: str):
        """Print streaming response in a chat-like format"""
        sys.stdout.write(content)
        sys.stdout.flush()

    def __call__(self, message: str) -> str:
        """Process user message and return response"""
        messages = self._prepare_messages(message)
        response = self.llm._make_request(messages, self.system_prompt)

        if self.verbose:
            print(f"\n{self.name} - User: ", message)
            print(f"\n{self.name} - Assistant: ", end="")

        if self.llm.stream:
            collected_messages = []
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        line = line[6:]
                        try:
                            data = json.loads(line)
                            if data["type"] == "content_block_delta":
                                content = data["delta"]["text"]
                                collected_messages.append(content)
                                if self.verbose:
                                    self._print_streaming_response(content)
                        except json.JSONDecodeError:
                            continue

            full_response = "".join(collected_messages)
            if self.verbose:
                print("\n")
        else:
            response_data = response.json()
            full_response = response_data["content"][0]["text"]
            if self.verbose:
                print(full_response + "\n")

        self.history.append({
            "role": "assistant",
            "content": full_response
        })
        self._save_conversation()

        return full_response
