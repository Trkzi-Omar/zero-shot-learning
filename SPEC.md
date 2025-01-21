# SPEC.md

## Overview
This document outlines the specification for creating and demonstrating three simple zero-shot learning (ZSL) applications. Each application showcases a different use case and uses pretrained models or embeddings to enable the recognition or classification of unseen classes.

---

## Text Sentiment Classification for Unseen Topics

### Objective
Implement a system that classifies the sentiment of text samples on topics not present in the training data. For example, if the training set is composed of movie reviews, the model should still be able to classify sentiment on sports or travel reviews without any labeled examples from those topics.

### Requirements
- **Model**: A text encoder or embedding model (e.g., Sentence-BERT or similar).
- **Training Data**: A labeled dataset of text samples for sentiment classification from one or two domains (e.g., movie reviews).
- **Test Data**: Unseen textual data (e.g., sports or travel reviews).
- **Infrastructure**: Python environment with libraries for embedding (e.g., Hugging Face Transformers).

### Implementation Outline
1. **Data Preparation**  
   - Collect or create a training set of labeled reviews from one domain.
   - Split the dataset into training and validation sets.
2. **Embedding Extraction**  
   - Encode training texts using a pretrained language model.
   - Encode each unseen category or domain label in a similar fashion.
3. **Training a Classifier**  
   - Train a model that maps text embeddings to sentiment labels (e.g., positive, negative).
   - Monitor performance on the validation set.
4. **Zero-Shot Inference**  
   - Use the same embedding model on unseen review texts.
   - Classify sentiment by comparing embeddings to known sentiment label representations or by feeding embeddings into the trained classifier.

### Deliverables
- A trained sentiment classification model.
- A script or function that takes new textual data (unseen domain) and outputs sentiment predictions.


---

## General Guidelines
- The three applications can be implemented independently.
- All rely on pretrained models or embeddings to manage unseen classes.
- Performance depends on the quality and relevance of the embedding model and textual or semantic descriptions.
- Each application should be accompanied by a simple test script or notebook to demonstrate zero-shot capabilities.

---

## Contact & Further Steps
- For additional support or clarifications, consult the documentation of the embedding models or frameworks mentioned.
- Upon completion, review results and refine the approach if performance is insufficient.
