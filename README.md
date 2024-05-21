## Medical Question Answering System

### Goal

The goal is to develop a medical question-answering system utilizing the provided dataset containing medical information.

### Approach Used to Tackle the Problem

The approach to tackle the problem involves building a simple sequence-to-sequence (Seq2Seq) model using an encoder-decoder architecture with LSTM layers with the help of Tensorflow and Keras.

### Assumptions Made During Model Development and Training

- Cleaned Text: Assumed that text data is preprocessed by converting to lowercase, removing special characters and digits, tokenizing, and removing stopwords.
  
- Word Tokenization: Assumed that questions and answers are tokenized at the word level using NLTK's `word_tokenize`.
  
- Padding: Assumed that sequences are padded to a fixed length (`max_seqlen` = 150) to handle variable input lengths.
  
- Evaluation Metrics: Assumed that using `sparse_categorical_crossentropy` loss and `accuracy` metric is appropriate for this language modeling task.
  - Validation Loss: This metric indicates how well the model is performing during training with respect to the loss function. Lower validation loss indicates better performance.
  - Validation Accuracy: This metric measures the proportion of correctly predicted words in the validation set. It helps to understand how accurate the model's predictions are.

### Model's Performance

- **Accuracy & Loss Function**:
  - Validation Accuracy: 0.6606
  - Validation Loss: 2.8565,
  - Early Stopping function was used to avoid overfitting.

- **Strengths**:
  - Can generate answers word by word based on input questions.
  - Handles sequence-to-sequence mapping effectively with LSTM layers.
  - Includes early stopping to prevent overfitting.
  
- **Weaknesses**:
  - Limited by the fixed sequence length (`max_seqlen` = 150) which might cut off longer answers.
  - May struggle with context and coherence in generating long answers.
  - May struggle in generating meaningful sequences word by word.

### Potential Improvements or Extensions

- Implementing attention mechanisms to focus on relevant parts of the input sequence.
- Using beam search during inference for better quality answers.
- Utilizing pretrained word embeddings like Word2Vec, GloVe, or FastText.
- Experimenting with more complex model architectures such as transformer-based models (like BERT).

Third-party LLM APIs (such as OpenAI, Claude, etc.) were not used for building this model and this model is built from scratch.
The dataset used is [intern_screening_dataset.csv](https://github.com/Hitesh-HYH/question-answering-system/files/15395809/intern_screening_dataset.csv)
