import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Dict, Union, Tuple
import numpy as np
from datasets import load_dataset
import logging
import re
from nltk.tokenize import word_tokenize
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZeroShotSentimentClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the zero-shot sentiment classifier.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Enhanced templates for better zero-shot performance
        self.sentiment_templates = {
            "positive": [
                "This text expresses a positive sentiment.",
                "The author is happy and satisfied.",
                "This review shows appreciation and approval.",
                "The tone of this text is optimistic and positive."
            ],
            "negative": [
                "This text expresses a negative sentiment.",
                "The author is dissatisfied and unhappy.",
                "This review shows criticism and disapproval.",
                "The tone of this text is pessimistic and negative."
            ],
            "neutral": [
                "This text expresses a neutral sentiment.",
                "The author is stating facts without emotion.",
                "This text is objective and unbiased.",
                "The tone of this text is balanced and neutral."
            ]
        }
        self.sentiment_embeddings = None
        self._compute_sentiment_embeddings()
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def _compute_sentiment_embeddings(self):
        """Compute embeddings for sentiment templates."""
        self.sentiment_embeddings = {}
        for sentiment, templates in self.sentiment_templates.items():
            # Average embeddings of all templates for each sentiment
            embeddings = [self.model.encode(template, convert_to_tensor=True) 
                         for template in templates]
            self.sentiment_embeddings[sentiment] = torch.mean(torch.stack(embeddings), dim=0)
    
    def predict(self, texts: Union[str, List[str]], return_confidence: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Predict sentiment for given text(s).
        
        Args:
            texts: Single text string or list of text strings
            return_confidence: Whether to return confidence scores
            
        Returns:
            If return_confidence is False: List of predicted sentiment labels
            If return_confidence is True: List of tuples (predicted_label, confidence_score)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Encode input texts
        text_embeddings = self.model.encode(processed_texts, convert_to_tensor=True)
        
        predictions = []
        for text_embedding in text_embeddings:
            # Compute cosine similarity with each sentiment embedding
            similarities = {
                sentiment: torch.nn.functional.cosine_similarity(
                    text_embedding.unsqueeze(0),
                    sentiment_embedding.unsqueeze(0)
                ).item()
                for sentiment, sentiment_embedding in self.sentiment_embeddings.items()
            }
            
            # Get sentiment with highest similarity and confidence score
            predicted_sentiment = max(similarities.items(), key=lambda x: x[1])
            
            if return_confidence:
                predictions.append((predicted_sentiment[0], predicted_sentiment[1]))
            else:
                predictions.append(predicted_sentiment[0])
            
        return predictions

    def visualize_predictions(self, texts: List[str], true_labels: List[str] = None):
        """
        Visualize prediction confidences for given texts.
        
        Args:
            texts: List of text samples
            true_labels: Optional list of true labels for comparison
        """
        predictions_with_conf = self.predict(texts, return_confidence=True)
        
        # Get confidence scores for all sentiments
        all_confidences = []
        for text in texts:
            text_embedding = self.model.encode(self.preprocess_text(text), convert_to_tensor=True)
            confidences = {
                sentiment: torch.nn.functional.cosine_similarity(
                    text_embedding.unsqueeze(0),
                    sentiment_embedding.unsqueeze(0)
                ).item()
                for sentiment, sentiment_embedding in self.sentiment_embeddings.items()
            }
            all_confidences.append(confidences)
        
        # Create heatmap
        confidence_matrix = np.array([[conf[sent] for sent in self.sentiment_templates.keys()] 
                                    for conf in all_confidences])
        
        plt.figure(figsize=(10, len(texts) * 0.5))
        sns.heatmap(confidence_matrix, 
                   annot=True, 
                   cmap='RdYlGn', 
                   xticklabels=list(self.sentiment_templates.keys()),
                   yticklabels=[text[:50] + '...' if len(text) > 50 else text for text in texts])
        plt.title('Sentiment Confidence Scores')
        plt.xlabel('Sentiment')
        plt.ylabel('Text')
        plt.tight_layout()
        return plt.gcf()

def evaluate_on_dataset(classifier: ZeroShotSentimentClassifier, 
                       texts: List[str], 
                       labels: List[str],
                       visualize: bool = True) -> Dict:
    """
    Evaluate the classifier on a dataset.
    
    Args:
        classifier: Initialized ZeroShotSentimentClassifier
        texts: List of text samples
        labels: List of true labels
        visualize: Whether to create visualization plots
        
    Returns:
        Dictionary containing evaluation metrics and optional plots
    """
    predictions = classifier.predict(texts)
    results = {
        'accuracy': accuracy_score(labels, predictions),
        'classification_report': classification_report(labels, predictions),
        'confusion_matrix': confusion_matrix(labels, predictions)
    }
    
    if visualize:
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, 
                   fmt='d',
                   xticklabels=list(classifier.sentiment_templates.keys()),
                   yticklabels=list(classifier.sentiment_templates.keys()))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        results['confusion_matrix_plot'] = plt.gcf()
        
        # Add confidence visualization for a subset of examples
        sample_size = min(10, len(texts))
        sample_indices = np.random.choice(len(texts), sample_size, replace=False)
        sample_texts = [texts[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]
        results['confidence_plot'] = classifier.visualize_predictions(sample_texts, sample_labels)
    
    return results 