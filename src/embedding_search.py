import pandas as pd
from pylate import indexes, models, retrieve
from typing import List, Tuple, Dict
import logging
from colorama import Fore, Style
import numpy as np
import os
import pickle

logger = logging.getLogger(__name__)

# Initialize the ColBERT model
model = models.ColBERT(
    model_name_or_path="jinaai/jina-colbert-v2",
    query_prefix="[QueryMarker]",
    document_prefix="[DocumentMarker]",
    attend_to_expansion_tokens=True,
    trust_remote_code=True,
    device="mps"
)

class EmbeddingSearcher:
    def __init__(self, index_folder: str = "pylate-index", index_name: str = "medical-classifications"):
        self.base_index_folder = index_folder
        self.base_index_name = index_name
        self.documents = []
        self.document_ids = []
        self.codes = []
        self.document_embeddings = None
        self.is_initialized = False
        self.embeddings_cache_path = os.path.join(index_folder, f"{index_name}_embeddings.pkl")

    def save_embeddings(self):
        """Save embeddings and related data to disk"""
        if not os.path.exists(self.base_index_folder):
            os.makedirs(self.base_index_folder)
        
        # Ensure embeddings are in the correct format before saving
        if not isinstance(self.document_embeddings, list):
            self.document_embeddings = [emb for emb in self.document_embeddings]
        
        cache_data = {
            'documents': self.documents,
            'document_ids': self.document_ids,
            'codes': self.codes,
            'document_embeddings': self.document_embeddings
        }
        
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Saved embeddings to {self.embeddings_cache_path}")

    def load_embeddings(self) -> bool:
        """Load embeddings from disk if they exist"""
        try:
            if os.path.exists(self.embeddings_cache_path):
                with open(self.embeddings_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.documents = cache_data['documents']
                self.document_ids = cache_data['document_ids']
                self.codes = cache_data['codes']
                self.document_embeddings = cache_data['document_embeddings']
                self.is_initialized = True
                logger.info(f"Loaded embeddings from {self.embeddings_cache_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False

    def initialize_with_dataframe(self, df: pd.DataFrame, force_recompute: bool = False):
        """Initialize with documents from a DataFrame"""
        # Try to load cached embeddings first
        if not force_recompute and self.load_embeddings():
            return

        self.documents = [f"{row['SLOVENSKI NAZIV']}" for _, row in df.iterrows()]
        self.document_ids = [str(i) for i in range(len(self.documents))]
        self.codes = [row['KODA'] for _, row in df.iterrows()]
        
        print("Encoding all documents...")
        # Store embeddings as a list to preserve per-token embeddings
        self.document_embeddings = [
            emb for emb in model.encode(
                self.documents,
                batch_size=128,
                is_query=False,
                show_progress_bar=True,
            )
        ]
        self.is_initialized = True
        
        # Save the embeddings for future use
        self.save_embeddings()
        print("Initialization complete")

    def create_subset_index(self, valid_indices: List[int], category: str) -> indexes.Voyager:
        """Create a new index for a subset of documents"""
        if not valid_indices:
            return None

        try:
            # Create subset of documents and embeddings
            subset_documents = [self.documents[i] for i in valid_indices]
            subset_document_ids = [str(i) for i in range(len(subset_documents))]
            
            # Handle ColBERT's per-token embeddings
            if isinstance(self.document_embeddings, list):
                # For ColBERT, each document has variable number of token embeddings
                subset_embeddings = [self.document_embeddings[i] for i in valid_indices]
            else:
                # If it's already a numpy array, just take the subset
                subset_embeddings = self.document_embeddings[valid_indices]

            logger.info(f"Creating index for {len(subset_documents)} documents")

            # Create new index for this category
            subset_index = indexes.Voyager(
                index_folder=self.base_index_folder,
                index_name=f"{self.base_index_name}-{category}",
                override=True  # Override any existing index for this category
            )

            # Add documents to subset index
            subset_index.add_documents(
                documents_ids=subset_document_ids,
                documents_embeddings=subset_embeddings,
            )

            return subset_index
        except Exception as e:
            logger.error(f"{Fore.RED}Error creating subset index: {str(e)}{Style.RESET_ALL}")
            return None

    def search_in_category(self, query: str, valid_indices: List[int], category_name: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """Search within specific document indices"""
        if not self.is_initialized:
            raise RuntimeError("Searcher not initialized. Call initialize_with_dataframe first.")

        try:
            if not valid_indices:
                logger.warning(f"{Fore.YELLOW}No valid indices provided for search{Style.RESET_ALL}")
                return []

            # Create mapping from subset index to original indices
            subset_to_original = {i: orig_i for i, orig_i in enumerate(valid_indices)}

            # Create subset index
            subset_index = self.create_subset_index(valid_indices, category_name)
            if not subset_index:
                return []

            # Create retriever for subset index
            subset_retriever = retrieve.ColBERT(index=subset_index)

            # Encode the query
            query_embedding = model.encode(
                [query],
                batch_size=1,
                is_query=True,
                show_progress_bar=False,
            )
            
            # Get results from subset index
            results = subset_retriever.retrieve(
                queries_embeddings=query_embedding,
                k=min(k, len(valid_indices)),
            )[0]  # Get first (and only) query results

            # Map results back to original documents
            filtered_results = []
            for res in results:
                subset_idx = int(res["id"])
                orig_idx = subset_to_original[subset_idx]
                filtered_results.append(
                    (self.documents[orig_idx], self.codes[orig_idx], res["score"])
                )

            return filtered_results
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error in search_in_category: {str(e)}{Style.RESET_ALL}")
            return []

# Global instance
searcher = EmbeddingSearcher() 