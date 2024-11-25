import json
import requests
import os
import faiss
import numpy as np
from typing import Optional, List, Union, Dict, Tuple
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class Voyage_Embedding:
    def __init__(
        self,
        model: str = "voyage-large-2",
        api_key: Optional[str] = None,
        index_directory: str = "faiss_indexes"
    ):
        self.model = model
        self.api_key = api_key or os.getenv('VOYAGE_API_KEY')
        self.index_directory = index_directory
        
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not found in environment variables")
        
        # Create necessary directories
        os.makedirs(self.index_directory, exist_ok=True)
        os.makedirs(f"{self.index_directory}/chunks", exist_ok=True)
        os.makedirs(f"{self.index_directory}/embeddings", exist_ok=True)

    def _make_request(self, input_text: Union[str, List[str]]) -> requests.Response:
        """Make request to Voyage Embeddings API"""
        url = "https://api.voyageai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Ensure input is a list
        if isinstance(input_text, str):
            input_text = [input_text]
        
        payload = {
            "model": self.model,
            "input": input_text
        }

        return requests.post(url, headers=headers, json=payload)

    def get_embedding(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Get embeddings for texts"""
        response = self._make_request(texts)
        
        if response.status_code != 200:
            raise Exception(f"Error in API call: {response.text}")
            
        result = response.json()
        
        # Extract embeddings from response
        embeddings = np.array([data["embedding"] for data in result["data"]], dtype=np.float32)
        
        # If single text, return single embedding
        if isinstance(texts, str):
            return embeddings[0]
        
        return embeddings

    def create_faiss_index(self, name: str, texts: List[str]) -> Tuple[faiss.Index, Dict]:
        """Create a FAISS index from a list of texts"""
        # Get embeddings
        embeddings = self.get_embedding(texts)
        
        # Ensure embeddings are in the correct shape and type
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Get actual dimension from embeddings
        actual_dim = embeddings.shape[1]
        
        # Create FAISS index with the actual dimension
        index = faiss.IndexFlatL2(actual_dim)
        index.add(embeddings)
        
        # Create chunks metadata
        chunks_metadata = {
            "created_at": datetime.now().isoformat(),
            "model": self.model,
            "total_chunks": len(texts),
            "embedding_dim": actual_dim,
            "chunks": [
                {
                    "id": i,
                    "text": text,
                    "embedding_index": i
                }
                for i, text in enumerate(texts)
            ]
        }
        
        # Save index and metadata
        self.save_index(name, index, chunks_metadata, embeddings)
        
        return index, chunks_metadata

    def save_index(self, name: str, index: faiss.Index, chunks_metadata: Dict, embeddings: np.ndarray):
        """Save FAISS index, chunks metadata, and embeddings"""
        # Save FAISS index
        faiss.write_index(index, f"{self.index_directory}/{name}.faiss")
        
        # Save chunks metadata
        with open(f"{self.index_directory}/chunks/{name}.json", 'w') as f:
            json.dump(chunks_metadata, f, indent=2)
        
        # Save embeddings
        np.save(f"{self.index_directory}/embeddings/{name}.npy", embeddings)

    def load_index(self, name: str) -> Tuple[faiss.Index, Dict, np.ndarray]:
        """Load FAISS index, chunks metadata, and embeddings"""
        # Load FAISS index
        index = faiss.read_index(f"{self.index_directory}/{name}.faiss")
        
        # Load chunks metadata
        with open(f"{self.index_directory}/chunks/{name}.json", 'r') as f:
            chunks_metadata = json.load(f)
        
        # Load embeddings
        embeddings = np.load(f"{self.index_directory}/embeddings/{name}.npy")
        
        return index, chunks_metadata, embeddings

    def search(self, name: str, query: str, k: int = 5) -> List[Dict]:
        """Search similar texts using FAISS"""
        # Load index and metadata
        index, chunks_metadata, _ = self.load_index(name)
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = index.search(query_embedding, k)
        
        # Get results with metadata
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            chunk = chunks_metadata["chunks"][idx]
            results.append({
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "distance": float(dist),
                "score": 1 / (1 + float(dist)),  # Convert distance to similarity score
                "rank": i + 1
            })
        
        return results

    def update_index(self, name: str, new_texts: List[str]):
        """Update existing index with new texts"""
        try:
            index, chunks_metadata, embeddings = self.load_index(name)
        except FileNotFoundError:
            print(f"Index {name} not found. Creating new index...")
            self.create_faiss_index(name, new_texts)
            return

        # Get embeddings for new texts
        new_embeddings = self.get_embedding(new_texts)
        
        # Update embeddings array
        embeddings = np.vstack([embeddings, new_embeddings])
        
        # Get dimension from embeddings
        actual_dim = embeddings.shape[1]
        
        # Update FAISS index
        index = faiss.IndexFlatL2(actual_dim)
        index.add(embeddings)
        
        # Update chunks metadata
        start_id = len(chunks_metadata["chunks"])
        new_chunks = [
            {
                "id": start_id + i,
                "text": text,
                "embedding_index": start_id + i
            }
            for i, text in enumerate(new_texts)
        ]
        chunks_metadata["chunks"].extend(new_chunks)
        chunks_metadata["total_chunks"] = len(chunks_metadata["chunks"])
        chunks_metadata["updated_at"] = datetime.now().isoformat()
        chunks_metadata["embedding_dim"] = actual_dim
        
        # Save updated index and metadata
        self.save_index(name, index, chunks_metadata, embeddings)
