import ktrain
from StandardiseDatasets import StandardiseDatasets
from Evaluate import evaluate

sd = StandardiseDatasets()
predictor = ktrain.load_predictor('models/tweeteval')
x, y_true = sd.get_TweetEval_test()

y_pred = predictor.predict(x)
y_pred = [False if i == 'not hate' else True for i in y_pred]

evaluate(y_true, y_pred)