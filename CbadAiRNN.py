from CbadAi import CbadAi
import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout, Embedding
from tensorflow.keras.metrics import Recall, Precision
from metrics import f1
import ktrain
from Evaluate import evaluate
from pathlib import Path

Path('logs/rnn').mkdir(parents=True, exist_ok=True)

class CbadAiRNN(CbadAi):   
    
    def train(self, dataset='all'):
        self.device_calibration()
            
        X_train, X_test, y_train, y_test, X, y = self._get_data(dataset, 'tensor')
            
        trn, val, preproc = ktrain.text.texts_from_array(X_train, y_train, X_test, y_test, maxlen=26)     
        
        
        model = tf.keras.Sequential([
            Embedding(30000, 15),
            Dropout(0.2),
            Bidirectional(LSTM(15)),
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
        
        print(model.summary())
        
        class validate(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                learner.validate(print_report=False, save_path='logs/rnn/e'+str(epoch+1)+'.csv', class_names=preproc.get_classes())
        
        learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=100)
        
        learner.model.compile(metrics=['accuracy'], loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam')
        
        learner.set_weight_decay(0.01)
        
        learner.fit_onecycle(1e-4, 20, callbacks=[
            tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    monitor='val_loss',
                    mode='min',
                    restore_best_weights=True
                    ),
            validate()
            ])
        
        self.y_train = y_train
        self.y_test = y_test
        self.X_test = X_test
        
        return model
    
    def test(self, dataset='all'):
        model = self.train(dataset=dataset)
        print(model.summary())
        #model.fit(data, epochs=3, batch_size=64)
        # Final evaluation of the model
        evaluate(self.y_true, self.y_pred)
