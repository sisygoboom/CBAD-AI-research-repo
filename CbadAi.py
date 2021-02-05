from sklearn.model_selection import train_test_split
from StandardiseDatasets import StandardiseDatasets
from sklearn.utils import Bunch
import matplotlib as plt
import tensorflow as tf
import numpy as np

class CbadAi:
    def __init__(self, test_size=0.05):
        
        self.ds = StandardiseDatasets()
        self.test_size = test_size
        
    def plot(self, history):
        for metric in ['loss', 'f1', 'precision', 'recall', 'accuracy']:
            plt.plot(history.history[metric])
            plt.plot(history.history['val_'+metric])
            plt.title('model '+metric)
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            
    def device_calibration(self):
        gpu = tf.config.experimental.list_physical_devices('GPU')
        if gpu:
            print('using gpu')
            #tf.config.experimental.set_memory_growth(gpu[0], True)
        else:
            print('using cpu')
    
    def _get_data(self, dataset, data_type='', string=True):
        
        switch = {
            'all': self.ds.get_all,
            #'maryland': self.ds.get_maryland,
            'cornell': self.ds.get_cornell,
            'tweeteval': self.ds.get_TweetEval_train
            }
        
        data = switch[dataset]()
        
        if not string:
            data['Code'] = [False if i == 'not hate' else True for i in data['Code']]
            
        
        print(data['Code'].value_counts())
        
        #data sanitization
        data['Tweet'] = data['Tweet'].replace('@\w{0,15}', '@user', regex=True)
        data['Tweet'] = data['Tweet'].replace(' ?(https:|http:|www\.)\S*? ', ' http ', regex=True)
        
        if data_type=='transformers':
            
            DATASET_SIZE = len(data)
            train_size = int(0.9 * DATASET_SIZE)
            val_size = int(0.1 * DATASET_SIZE)
            
            y = (data['Code'].values)
            x = list(data['Tweet'].values)
            
            full_dataset = tf.data.Dataset.from_tensor_slices((x, y))
            full_dataset = full_dataset.shuffle(10000)
            
            train_dataset = full_dataset.take(train_size)
            val_dataset = full_dataset.skip(val_size)
            
            return train_dataset, val_dataset
            
        
        if data_type=='tensor':
            X_train, X_test, y_train, y_test = train_test_split(
                data['Tweet'].values, 
                data['Code'].values,
                test_size=self.test_size,
                random_state=200
                )
            
            
            
            return X_train, X_test, y_train, y_test, data['Tweet'].values, data['Code'].values
            
            
        if data_type=='ktrain':
            X_train, X_test, y_train, y_test = train_test_split(
                data.Tweet.values, 
                data.Code.values, 
                test_size=self.test_size,
                stratify=data['Code'],
                random_state=200
                )
            
            return Bunch(data=X_train, target=y_train), Bunch(data=X_test, target=y_test)
            
        else:
            return train_test_split(
                data.Tweet, 
                data.Code, 
                test_size=self.test_size,
                stratify=data['Code'],
                random_state=200
                )



