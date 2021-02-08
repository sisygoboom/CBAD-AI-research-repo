from CbadAi import CbadAi
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, Input, Dropout
from Evaluate import evaluate
import ktrain



class CbadAiCNN(CbadAi):    
    
    def train(self, dataset='all', char=False):
        
        self.device_calibration()
            
        X_train, X_test, y_train, y_test, X, y = self._get_data(dataset, 'tensor', False)
            
        
        trn, val, preproc = ktrain.text.texts_from_array(X_train, y_train, X_test, y_test, maxlen=26)
        
        
        if char:
            model = tf.keras.Sequential([
                Input(shape=(280,), dtype='int32'),
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
                Input(shape=(26,), dtype='int32'),
                Embedding(30000, 10),
                Dropout(0.5),
                Conv1D(6, 2, activation='relu'),
                MaxPooling1D(2),
                Conv1D(3, 2, activation='relu'),
                MaxPooling1D(11),
                Dense(1, activation='sigmoid')
            ])
        
        print(model.summary())
        
        class validate(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                learner.validate(print_report=False, save_path='logs/cnn/e'+str(epoch+1)+'.csv', class_names=preproc.get_classes())
        
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
        
        #self.plot(history)
        
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return model
    
    def test(self, dataset='all', char=False):
        model = self.train(dataset=dataset, char=char)
        print(model.summary())
        y_pred = model.predict(self.X_test)
        y_true = self.y_test
        
        evaluate(y_true, y_pred)
