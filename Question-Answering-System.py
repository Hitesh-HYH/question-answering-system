#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing all the libraries required
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


# Load the dataset
dataset = pd.read_csv('intern_screening_dataset.csv')


# In[ ]:


# Converting the dataset into strings
dataset['question'] = dataset['question'].astype(str)
dataset['answer'] = dataset['answer'].astype(str)


# In[ ]:


# Downloading necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# In[ ]:


def clean_text(text):
    # Lowercasing the text
    text = text.lower()
    # Removing special characters and digits
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenizing the data
    tokens = word_tokenize(text)
    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# In[ ]:


# Applying the cleaning function to the dataset
dataset['question'] = dataset['question'].apply(clean_text)
dataset['answer'] = dataset['answer'].apply(clean_text)


# In[ ]:


# Tokenizing the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['question'].values)
questions_seq = tokenizer.texts_to_sequences(dataset['question'].values)
answers_seq = tokenizer.texts_to_sequences(dataset['answer'].values)


# In[ ]:


# Pad the sequences
max_seqlen = 150
questions_padded = pad_sequences(questions_seq, maxlen=max_seqlen)
answers_padded = pad_sequences(answers_seq, maxlen=max_seqlen)


# In[ ]:


# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(questions_padded, answers_padded, test_size=0.2, random_state=42)


# In[ ]:


# Define the model architecture
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
hidden_units = 128


# In[ ]:


# Encoder
encoder_inputs = Input(shape=(max_seqlen,))
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(hidden_units)(encoder_embedding)


# In[ ]:


# Decoder
decoder_inputs = RepeatVector(max_seqlen)(encoder_lstm)
decoder_lstm = LSTM(hidden_units, return_sequences=True)(decoder_inputs)
decoder_dense = Dense(vocab_size, activation='softmax')(decoder_lstm)


# In[ ]:


# Model
model = Model(encoder_inputs, decoder_dense)


# In[ ]:


# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()


# In[ ]:


# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# In[ ]:


# Train the model
epochs = 20
batch_size = 32
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])


# In[ ]:


# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')


# In[ ]:


# Function to generate answers
def generate_answer(question):
    question_seq = tokenizer.texts_to_sequences([question])
    question_padded = pad_sequences(question_seq, maxlen=max_seqlen)
    print(f'Question sequence before padding: {question_seq}')
    print(f'Padded question sequence: {question_padded}')
    answer_seq = model.predict(question_padded)
    print(f'Answer sequence predicted: {answer_seq}')
    answer = ' '.join([tokenizer.index_word[word] for word in np.argmax(answer_seq, axis=-1)[0] if word != 0])
    return answer

# Example questions
questions = [
    "What causes heart failure?",
    "How to prevent Bronchitis?",
    "What is Kidney Disease?"
]

# Generate answers
for question in questions:
    generated_answer = generate_answer(question)
    print(f'Question: {question}')
    print(f'Generated Answer: {generated_answer}\n')

