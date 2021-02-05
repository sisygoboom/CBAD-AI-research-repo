from CbadAi import CbadAi
import tensorflow as tf
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout, Embedding
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from metrics import f1


class CbadAiRNN(CbadAi):   
    
    def train(self, dataset='all'):
        self.device_calibration()
            
        X_train, X_test, y_train, y_test, X, y = self._get_data(dataset, True)
            
        vocab_size = 20000 # make the top list of words (common words)
        embedding_dim = 10
        max_length = 50
        trunc_type = 'post'
        padding_type = 'post'
        oov_tok = '<OOV>' # OOV = Out of Vocabulary
        
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok, lower=True, char_level=False)
        tokenizer.fit_on_texts(X)
        word_index = tokenizer.word_index
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        
        
        
        model = tf.keras.Sequential([
            Embedding(vocab_size, embedding_dim),
            Dropout(0.2),
            Bidirectional(LSTM(embedding_dim)),
            Dense(1,activation='sigmoid')
        ])
        
        print(model.summary())
        
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=[
                          f1,
                          Recall(name='recall'),
                          Precision(name='precision'),
                          'accuracy'
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
                    patience=10,
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
    
    def test(self, dataset='all'):
        model = self.train(dataset=dataset)
        print(model.summary())
        #model.fit(data, epochs=3, batch_size=64)
        # Final evaluation of the model
        scores = model.evaluate(self.X_test_padded, self.y_test)
        print(scores)
