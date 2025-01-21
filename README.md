# Zero-Shot Sentiment Analysis

This project implements a zero-shot learning approach to sentiment analysis, capable of classifying text sentiment across various domains without requiring domain-specific training data.

## Project Structure

```
.
├── SPEC.md              # Project specifications and requirements
├── requirements.txt     # Python dependencies
├── src/
│   └── sentiment/
│       ├── __init__.py     # Package initialization
│       └── classifier.py   # Core sentiment classifier implementation
├── notebooks/
│   └── sentiment_analysis_demo.ipynb  # Interactive demo notebook
└── data/                # Directory for datasets (if needed)
```

## Components

### Core Implementation (`src/sentiment/`)

- `classifier.py`: Implements the `ZeroShotSentimentClassifier` class, which uses pre-trained language models (Sentence-BERT) to perform zero-shot sentiment classification. Features include:
  - Text preprocessing
  - Sentiment prediction with confidence scores
  - Visualization tools
  - Batch processing capabilities
  - Performance evaluation metrics

### Demo Notebook (`notebooks/sentiment_analysis_demo.ipynb`)

An interactive Jupyter notebook that demonstrates the classifier's capabilities:
1. Single text analysis
2. Multi-domain sentiment classification
3. Performance evaluation on IMDB dataset
4. Interactive demo with real-time analysis
5. Visualization of confidence scores and predictions

## Features

- **Zero-shot Learning**: Works on any domain without specific training
- **Confidence Scores**: Provides confidence levels for predictions
- **Multi-domain Support**: Works across various text types (reviews, comments, etc.)
- **Visualization Tools**: Includes heatmaps and confidence plots
- **Batch Processing**: Handles both single texts and large datasets
- **Interactive Demo**: Try the classifier with your own text

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd zero-shot-learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Using the Classifier

```python
from src.sentiment import ZeroShotSentimentClassifier

# Initialize the classifier
classifier = ZeroShotSentimentClassifier()

# Single text prediction
text = "This product exceeded my expectations!"
sentiment, confidence = classifier.predict(text, return_confidence=True)[0]
print(f"Sentiment: {sentiment}, Confidence: {confidence:.3f}")

# Batch prediction
texts = ["Great service!", "Poor quality", "Neutral statement"]
predictions = classifier.predict(texts)
```

### Running the Demo

1. Start Jupyter notebook:
```bash
jupyter notebook
```

2. Open `notebooks/sentiment_analysis_demo.ipynb`
3. Follow the interactive demonstrations

## Dependencies

- torch
- sentence-transformers
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- nltk
- datasets
- ipywidgets (for interactive demo)

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
