# rerank.py
from sentence_transformers import CrossEncoder
from typing import List, Tuple # Added for type hinting

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the Reranker with a CrossEncoder model.

        Args:
            model_name (str): The name of the CrossEncoder model to load
                              from sentence-transformers.
        """
        try:
            self.model = CrossEncoder(model_name)
            print(f"CrossEncoder model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading CrossEncoder model '{model_name}': {e}")
            self.model = None

    def cross_with_indices(self, query: str,
                           candidate_indices: List[int],
                           candidate_texts: List[str]) -> List[Tuple[int, float]]:
        """
        Re-ranks a list of candidate texts based on their relevance to a query,
        preserving their original indices.

        Args:
            query (str): The input query string.
            candidate_indices (List[int]): A list of original indices corresponding
                                           to the candidate texts.
            candidate_texts (List[str]): A list of candidate texts to be re-ranked.

        Returns:
            List[Tuple[int, float]]: A list of (index, score) tuples, sorted by
                                     score in descending order. Returns an empty
                                     list if the model is not loaded or inputs
                                     are problematic.
        """
        if self.model is None:
            print("Warning: Reranker model not loaded. Cannot perform re-ranking.")
            # Return original indices with a dummy score or handle as an error
            return sorted([(idx, 0.0) for idx in candidate_indices], key=lambda x: x[0])


        if not candidate_texts or not candidate_indices:
            # print("Warning: Empty candidate texts or indices provided to reranker.")
            return []

        if len(candidate_indices) != len(candidate_texts):
            print("Error: candidate_indices and candidate_texts must have the same length.")
            # Fallback: return original indices with a dummy score, sorted by index
            return sorted([(idx, 0.0) for idx in candidate_indices], key=lambda x: x[0])

        # Create pairs of [query, candidate_text] for the model
        pairs = [[query, text] for text in candidate_texts]

        try:
            # Get scores from the CrossEncoder model
            # Set show_progress_bar=False for cleaner logs during bulk processing
            scores = self.model.predict(pairs, show_progress_bar=False)

            # Ensure scores is a list of floats if it's a numpy array
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()

            # Combine original indices with their new scores
            ranked_with_indices = sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)
            return ranked_with_indices
        except Exception as e:
            print(f"Error during CrossEncoder prediction: {e}")
            # Fallback: return original indices with a dummy score, sorted by index
            return sorted([(idx, 0.0) for idx in candidate_indices], key=lambda x: x[0])

