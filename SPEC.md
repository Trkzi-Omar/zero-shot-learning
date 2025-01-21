# SPEC.md

## Overview
This document outlines the specification for creating and demonstrating three simple zero-shot learning (ZSL) applications. Each application showcases a different use case and uses pretrained models or embeddings to enable the recognition or classification of unseen classes.

---

## 1. Text Sentiment Classification for Unseen Topics

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

## 2. Image Classification with Zero-Shot Labels

### Objective
Classify images from classes not present in the training data using text descriptions or labels. For instance, if the training images involve dogs and cats, the system should detect zebras or lions by relying on textual embeddings.

### Requirements
- **Model**: A vision-language model such as CLIP.
- **Training Data**: A small labeled dataset of seen classes (optional if using a fully pretrained vision-language model).
- **Unseen Class Descriptions**: Textual or semantic descriptions of new image classes.
- **Infrastructure**: Python environment with PyTorch or similar frameworks.

### Implementation Outline
1. **Data Preparation**  
   - Gather a few labeled images for seen classes (if fine-tuning is necessary).
   - Collect images for testing from unseen categories.
2. **Model Setup**  
   - Use a pretrained CLIP model (or equivalent) that generates embeddings for both images and text.
3. **Zero-Shot Inference**  
   - Encode the unseen class descriptions into text embeddings.
   - Encode the test images into image embeddings.
   - Match image embeddings to the closest text embedding using similarity measures.
4. **Evaluation**  
   - Compare zero-shot predictions to actual labels of the unseen classes.
   - Measure accuracy or other metrics to validate performance.

### Deliverables
- A script that, given an image and textual descriptions of unseen classes, returns the most likely class.
- A small test dataset of unseen categories to validate performance.

---

## 3. Intent Recognition for Chatbots

### Objective
Enable a chatbot to recognize user intents not included in the original training set. For example, if the chatbot was trained on "play music" and "check weather," it should identify new intents such as "book a flight."

### Requirements
- **Model**: A sentence embedding model (e.g., Universal Sentence Encoder, Sentence-BERT).
- **Training Data**: A set of sample queries labeled with known intents.
- **Unseen Intent Descriptions**: Descriptions for new intents to be recognized at inference time.
- **Infrastructure**: Python environment with relevant NLP libraries.

### Implementation Outline
1. **Data Preparation**  
   - Collect a dataset of user queries mapped to known intents (e.g., "play music," "check weather").
2. **Embedding Extraction**  
   - Encode the training queries and their intent labels using a sentence embedding model.
   - Encode each unseen intent’s description in the same embedding space.
3. **Intent Classifier**  
   - Train a classifier or use a similarity-based approach to match user queries to the closest known intent embedding.
4. **Zero-Shot Recognition**  
   - When encountering a query for an unseen intent, compare the query embedding to the embedding of the new intent description.
   - Assign the new intent if it has the highest similarity.

### Deliverables
- A trained intent recognition module.
- A process for adding and testing new intents without retraining on labeled samples.

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
