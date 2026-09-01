import os
import asyncio
import chromadb
from chromadb.config import Settings
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any

from .config import settings
from .exceptions import DatabaseConnectionError

class AsyncChromaManager:
    """
    Asynchronous interface for ChromaDB vector operations.
    Complies with the strict non-blocking I/O requirement.
    """
    def __init__(self, collection_name: str = "symantix_logs"):
        try:
            self.client = chromadb.AsyncHttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=Settings(allow_reset=True)
            )
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to connect to ChromaDB: {e}")
            
        self.collection_name = collection_name
        self.collection = None
        
        # Load the custom fine-tuned embedding model
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core-ml", "weights")
        if not os.path.exists(model_path):
            # Fallback to the base model if not yet trained/available
            model_path = "BAAI/bge-small-en-v1.5"
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    async def initialize_collection(self):
        """Asynchronously initialize or get the vector collection."""
        self.collection = await self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous embedding generation (CPU/GPU bound). 
        Should ideally be run in a thread pool for true non-blocking API integration.
        """
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Pillar A: Zero-Hallucination & Math Verification
        assert sentence_embeddings.shape[1] == 384, f"Embedding dimension mismatch. Expected 384, got {sentence_embeddings.shape[1]}"
        
        # Explicit memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return sentence_embeddings.cpu().tolist()

    async def ingest_logs(self, log_ids: List[str], log_messages: List[str], metadata_list: List[Dict[str, Any]]):
        """
        Asynchronously embed and upsert logs to ChromaDB.
        """
        if not self.collection:
            await self.initialize_collection()
            
        assert self.collection is not None, "Collection failed to initialize"
            
        # Run CPU/GPU bound embedding generation in a separate thread to prevent event loop blocking
        embeddings = await asyncio.to_thread(self.generate_embeddings, log_messages)
        
        try:
            await self.collection.upsert(
                documents=log_messages,
                embeddings=embeddings,
                metadatas=metadata_list,
                ids=log_ids
            )
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to upsert vectors to ChromaDB: {e}")

    async def retrieve_similar(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve semantically similar logs from the vector store.
        """
        if not self.collection:
            await self.initialize_collection()
            
        assert self.collection is not None, "Collection failed to initialize"
            
        query_embedding = await asyncio.to_thread(self.generate_embeddings, [query])
        
        try:
            results = await self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            return results
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to query vectors from ChromaDB: {e}")
