import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                'Logistic Regression': LogisticRegression(),
                'Decision Tree': DecisionTreeClassifier(),
                'Support Vector Classifier': SVC(),
                'Naive Bayes': GaussianNB(),
                'Random Forest Classifier': RandomForestClassifier(),
                'AdaBoost Classifier': AdaBoostClassifier(),
                'Gradient Boosting Classifier': GradientBoostingClassifier(),
                'XGBClassifier': XGBClassifier(),
                'K Neighbors': KNeighborsClassifier(),
                'CatBoosting Classifier': CatBoostClassifier()
            }

            params = {
                'Logistic Regression': {},
                'Decision Tree': {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [1,2,3,4,6,8],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2']
                },
                'Support Vector Classifier': {
                    'C': [1,2,3,4,5,10,20, 50],
                    'gamma': [0.2, 1, 0.4, 0.001, 0.003],
                    'kernel': ['linear']
                },
                'Naive Bayes': {},
                'Random Forest Classifier': {
                    'max_depth': [1,2,3,4, 10,20,30],
                    'n_estimators': [50,60,70, 80, 90,100],
                    'criterion': ['gini', 'entropy']
                },
                'AdaBoost Classifier': {
                    'n_estimators': [50,100,200],
                    'learning_rate': [0.001, 0.1, 1, 1.5,2],
                    'algorithm': ['SAMME.R', 'SAMME']
                },
                'Gradient Boosting Classifier': {
                    'n_estimators': [50,100,200],
                    'learning_rate': [0.001, 0.1, 1,1.5,2,2.5],
                    'ccp_alpha': [1,2]
                },
                'XGBClassifier': {
                    'n_estimators': [50,80,100],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2]
                },
                'K Neighbors': {
                    "n_neighbors": [3, 5, 6, 7, 10, 12, 15],
                    'algorithm': ['ball_tree', 'brute', 'kd_tree'],
                    'leaf_size': [20, 30, 40 , 50]
                },
                'CatBoosting Classifier': {}
            }

            
            model_report: dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            logging.info(f'best found model on both training and testing dataset.')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model_name
            )

            predicted = best_model.predict(X_test)

            perfect_score = accuracy_score(y_test, predicted)
            return perfect_score
        
        except Exception as e:
            raise CustomException(e, sys)