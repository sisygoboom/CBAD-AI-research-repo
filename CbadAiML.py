from CbadAi import CbadAi
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# from sklearn.linear_model import SGDClassifier, LogisticRegression
# from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier
# from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.ensemble import VotingClassifier, BaggingClassifier
# from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
# from sklearn.neighbors import KNeighborsTransformer
from sklearn.svm import SVC, LinearSVC
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class CbadAiML(CbadAi):
    def train(self, dataset='all', X_train=None, y_train=None):
        
        if (X_train is None or y_train is None):
            print('training data not supplied')
            X_train, X_test, y_train, y_test = self._get_data(dataset)
        
        text_pipe = Pipeline([
            ('vect', TfidfVectorizer(analyzer='word')),
            ('tfidf', TfidfTransformer(use_idf=True)),
            #('tfidf', KNeighborsTransformer(mode='connectivity', leaf_size=0.0001, metric='cityblock')),
            #('clf', KNeighborsClassifier())
            #('clf', RidgeClassifier())
            #('clf', ComplementNB())
            #('clf', BernoulliNB())
            #('clf', BaggingClassifier())
            # ('clf', VotingClassifier(estimators=[
            #     ('lsvm', LinearSVC()),
            #     ('pa', PassiveAggressiveClassifier()),
            #     ('b', BaggingClassifier()),
            #     ('r', RidgeClassifier())
            # ])),
            #('clf', MLPClassifier())
            #('clf', PassiveAggressiveClassifier(C=0.1, average=True, class_weight='balanced',shuffle=False, warm_start=True))
            #('clf', SGDClassifier())
            #('clf', KNeighborsClassifier()),
            #('clf', RandomForestClassifier()),
            #('clf', MultinomialNB()),
            #('clf', LogisticRegression()),
            #('clf', DecisionTreeClassifier(class_weight='balanced', splitter='random'))
            #('clf', AdaBoostClassifier())
            # ('clf', SVC(
            #     kernel='linear',
            #     C=6,
            #     degree=1, 
            #     coef0=-1, 
            #     shrinking=False, 
            #     probability=True,
            #     class_weight='balanced',
            #     decision_function_shape='ovo'
            #     ))
            ('clf', LinearSVC(class_weight='balanced', loss='hinge', C=0.1))
            #('clf', NearestCentroid())
            #('clf', ExtraTreeClassifier())
            #('clf', GradientBoostingClassifier())
        ])
        
        text_pipe.fit(X_train, y_train)
        
        return text_pipe
    
    
    def test(self, dataset='all'):
        
        X_train, X_test, y_train, y_test = self._get_data(dataset)
        
        text_pipe = self.train(dataset, X_train, y_train)
        
        grid_params = {
            #'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            #'clf__C': [6,7,8],
            #'clf__degree': [1],
            #'clf__gamma': ['scale', 'auto'],
            #'clf__coef0': [-1, -2, 0, 1, 2],
            #'clf__shrinking': (True, False),
            #'clf__probability': (True, False),
            #'clf__class_weight': ['balanced', None],
            #'clf__decision_function_shape': ['ovo', 'ovr'],
            #'clf__break_ties': (True, False)
        }
        
        search = GridSearchCV(text_pipe, grid_params, scoring='f1_macro')
        search.fit(X_train, y_train)
        
        return {
            'best_estimator': search.best_estimator_,
            'error_score': search.error_score,
            'score': search.score(X_test, y_test),
            'best_score': search.best_score_
        }