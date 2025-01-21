from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZeroShotSentimentClassifier:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the zero-shot sentiment classifier.
        
        Args:
            model_name (str): Name of the pretrained Sentence-BERT model to use
        """
        self.model = SentenceTransformer(model_name)
        self.sentiment_templates = {
            'positive': 'This text expresses a positive sentiment.',
            'negative': 'This text expresses a negative sentiment.',
            'neutral': 'This text expresses a neutral sentiment.'
        }
        self._compute_sentiment_embeddings()
    
    def _compute_sentiment_embeddings(self):
        """Compute embeddings for sentiment templates."""
        self.sentiment_embeddings = {
            sentiment: self.model.encode(template, convert_to_tensor=True)
            for sentiment, template in self.sentiment_templates.items()
        }
    
    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict sentiment for given text(s).
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Predicted sentiment(s)
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
            
        # Encode input texts
        text_embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        # Calculate similarities with sentiment templates
        predictions = []
        for text_embedding in text_embeddings:
            similarities = {
                sentiment: cosine_similarity(
                    text_embedding.cpu().numpy().reshape(1, -1),
                    sentiment_embedding.cpu().numpy().reshape(1, -1)
                )[0][0]
                for sentiment, sentiment_embedding in self.sentiment_embeddings.items()
            }
            predicted_sentiment = max(similarities.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_sentiment)
        
        return predictions[0] if single_input else predictions
    
    def predict_with_scores(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Predict sentiment with confidence scores for given text(s).
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Dictionary or list of dictionaries containing predictions and scores
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
            
        # Encode input texts
        text_embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        # Calculate similarities with sentiment templates
        results = []
        for text_embedding in text_embeddings:
            similarities = {
                sentiment: float(cosine_similarity(
                    text_embedding.cpu().numpy().reshape(1, -1),
                    sentiment_embedding.cpu().numpy().reshape(1, -1)
                )[0][0])
                for sentiment, sentiment_embedding in self.sentiment_embeddings.items()
            }
            # Normalize scores using softmax
            scores = np.array(list(similarities.values()))
            exp_scores = np.exp(scores)
            softmax_scores = exp_scores / exp_scores.sum()
            
            results.append({
                'prediction': max(similarities.items(), key=lambda x: x[1])[0],
                'scores': dict(zip(similarities.keys(), softmax_scores))
            })
        
        return results[0] if single_input else results 