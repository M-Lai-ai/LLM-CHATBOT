import json
import requests
import os
from typing import Optional, List, Dict, Union, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Voyage_Rerank:
    def __init__(
        self,
        model: str = "rerank-2",
        api_key: Optional[str] = None,
        top_k: int = 3
    ):
        self.model = model
        self.top_k = top_k
        self.api_key = api_key or os.getenv('VOYAGE_API_KEY')
        
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not found in environment variables")

    def _make_request(self, query: str, documents: List[str]) -> requests.Response:
        """Make request to Voyage Rerank API"""
        url = "https://api.voyageai.com/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "documents": documents,
            "model": self.model,
            "top_k": self.top_k,
            "return_documents": True
        }

        return requests.post(url, headers=headers, json=payload)

    def rerank(self, query: str, documents: List[str]) -> List[Dict]:
        """Rerank documents based on query"""
        if not documents:
            return []
            
        response = self._make_request(query, documents)
        
        if response.status_code != 200:
            raise Exception(f"Error in API call: {response.text}")
            
        result = response.json()
        
        # Return reranked results with scores and original documents
        reranked_results = []
        for item in result["data"]:
            reranked_results.append({
                "text": item["document"],
                "score": item["relevance_score"],
                "original_index": item["index"]
            })
            
        return reranked_results

    def get_best_chunks(self, query: str, documents: List[str]) -> List[str]:
        """Get only the best chunks after reranking"""
        reranked = self.rerank(query, documents)
        return [item["text"] for item in reranked]
