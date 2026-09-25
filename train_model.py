import numpy as np
import nltk
from nltk.corpus import gutenberg
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 1. Data Collection
nltk.download('gutenberg')
data = gutenberg.raw('shakespeare-hamlet.txt')

with open('hamlet.txt', 'w') as file:
    file.write(data)

# 2. Data Preprocessing
with open('hamlet.txt', 'r') as file:
    text = file.read().lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

x, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. Model Building (LSTM)
model = Sequential()
model.add(tf.keras.Input(shape=(max_sequence_len - 1,), dtype="int32"))
model.add(Embedding(input_dim=total_words, output_dim=100))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.20))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Model Training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])

# 5. Saving Artifacts
model.save("next_word_lstm.h5")

with open("tokenizer.pickle", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model aur Tokenizer successfully save ho gaye hain!")