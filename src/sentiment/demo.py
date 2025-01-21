from sentiment_classifier import ZeroShotSentimentClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize classifier
    logger.info("Initializing Zero-Shot Sentiment Classifier...")
    classifier = ZeroShotSentimentClassifier()
    
    # Test cases from different domains
    test_cases = {
        "Movie Review": "The film was a masterpiece with stunning visuals and an engaging plot.",
        "Restaurant Review": "The food was cold and the service was terrible.",
        "Product Review": "This laptop works fine, it gets the job done but nothing special.",
        "Sports Commentary": "An incredible victory that will be remembered for years to come!",
        "Tech News": "The latest software update has caused significant issues for users."
    }
    
    logger.info("\nTesting Zero-Shot Sentiment Classification across domains:")
    print("\n" + "="*50)
    
    for domain, text in test_cases.items():
        result = classifier.predict_with_scores(text)
        
        print(f"\nDomain: {domain}")
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {result['prediction']}")
        print("Confidence Scores:")
        for sentiment, score in result['scores'].items():
            print(f"  {sentiment}: {score:.3f}")
        print("-"*50)

if __name__ == "__main__":
    main() 