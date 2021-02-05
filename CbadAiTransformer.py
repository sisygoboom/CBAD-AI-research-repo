import ktrain
from ktrain import text
from CbadAi import CbadAi
import tensorflow as tf
from metrics import f1
from pathlib import Path

cb = CbadAi(test_size=2)
#cb = CbadAi()

train_b, test_b = cb._get_data('tweeteval', 'ktrain')
#train_b, test_b = cb._get_data('all', 'ktrain')

x_train = train_b.data
y_train = train_b.target
x_test = test_b.data
y_test = test_b.target

model_name = 'vinai/bertweet-base'
lr = 3e-5
wd = 0.1
log_path = 'logs/'+model_name+'/lr_'+str(lr)+'/wd_'+str(wd)
chk_path = 'models/tweeteval'
#chk_path = 'models/'+ model_name+'-lr_'+str(lr)+'-wd_'+str(wd)

Path(log_path).mkdir(parents=True, exist_ok=True)

t = text.Transformer(model_name, maxlen=26)

trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)

model = t.get_classifier(multilabel=False, metrics=[
    'accuracy',
    f1
    ])

learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=100)


class validate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        learner.validate(print_report=False, save_path=log_path+'/e'+str(epoch+1)+'.csv', class_names=t.get_classes())

learner.set_weight_decay(wd)

learner.fit_onecycle(
    lr, 
    2, 
    checkpoint_folder='models/'+ model_name + '-lr_'  + str(lr) + '-wd_' + str(wd),
    callbacks=[
        validate()
    ])

predictor = ktrain.get_predictor(learner.model, preproc=t)

predictor.save(chk_path)