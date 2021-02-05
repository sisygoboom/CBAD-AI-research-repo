from CbadAi import CbadAi
import tensorflow as tf
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, Input, Flatten, Dropout
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
#from metrics import f1
from tensorflow.keras.metrics import Recall, Precision

import keras.backend as K

def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

class CbadAiCNN(CbadAi):    
    
    def train(self, dataset='all', char=False):
        
        self.device_calibration()
            
        X_train, X_test, y_train, y_test, X, y = self._get_data(dataset, True)
        
        if char:
            vocab_size = 50 # make the top list of words (common words)
            embedding_dim = 15
            max_length = 280
            trunc_type = 'post'
            padding_type = 'post'
            oov_tok = '<OOV>' # OOV = Out of Vocabulary
            
            tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok, lower=False, char_level=True)
            
        if not char:
            vocab_size = 100000 # make the top list of words (common words)
            embedding_dim = 10
            max_length = 50
            trunc_type = 'post'
            padding_type = 'post'
            oov_tok = '<OOV>' # OOV = Out of Vocabulary
            
            tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok, lower=True)
            
            
        tokenizer.fit_on_texts(X)
        word_index = tokenizer.word_index
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        
        
        if char:
            model = tf.keras.Sequential([
                Input(shape=(max_length,), dtype='int32'),
                Embedding(vocab_size, embedding_dim),
                Dropout(0.5),
                Conv1D(70, 2, activation='relu'),
                MaxPooling1D(2),
                Conv1D(70, 2, activation='relu'),
                MaxPooling1D(2),
                Conv1D(70, 2, activation='relu'),
                MaxPooling1D(2),
                GlobalMaxPooling1D(),
                Dense(1, activation='sigmoid')
            ])
        if not char:
            model = tf.keras.Sequential([
                Input(shape=(max_length,), dtype='int32'),
                Embedding(vocab_size, embedding_dim),
                Dropout(0.5),
                Conv1D(70, 2, activation='relu'),
                MaxPooling1D(2),
                Conv1D(70, 2, activation='relu'),
                MaxPooling1D(15),
                Dense(1, activation='sigmoid')
            ])
        
        print(model.summary())
        
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=[
                          f1,
                          'accuracy',
                          Recall(name='recall'),
                          Precision(name='precision')
                          ]
                      )
        
        num_epochs = 400
        
        history = model.fit(
            X_train_padded, 
            y_train, 
            epochs=num_epochs, 
            validation_data=(X_test_padded, y_test),
            verbose=2,
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=50,
                    monitor='val_f1',
                    mode='max',
                    restore_best_weights=True
                    )
                ]
            )
        
        self.plot(history)
        
        self.X_test_padded = X_test_padded
        self.X_train_padded = X_train_padded
        self.y_train = y_train
        self.y_test = y_test
        
        return model
    
    def test(self, dataset='all', char=False):
        model = self.train(dataset=dataset, char=char)
        print(model.summary())
        #model.fit(data, epochs=3, batch_size=64)
        # Final evaluation of the model
        scores = model.evaluate(self.X_test_padded, self.y_test)
        print(scores)
