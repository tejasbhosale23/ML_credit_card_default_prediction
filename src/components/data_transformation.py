import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            df1 = pd.read_csv('artifacts/train.csv')
            df = df1.drop('default.payment.next.month', axis=1)

            cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']

            num_features = [feature for feature in df.columns if df[feature].dtype != 'O']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encode', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical columns: {cat_features}')
            logging.info(f'numerical columns: {num_features}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('obtaining prprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'default.payment.next.month'

            # df = pd.read_csv('artifacts/train.csv')
            # numerical_column = [feature for feature in df if df[feature].dtype != 'O']


            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info(f'Applying preprocessing object on training and testing dataframe')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

# np.c_ - This is a special object in numpy that allows for concatenation
#         along the second axis.

# train_arr = concatenates 'input_feature_train_arr' and numpy array version of 
#             'target_feature_train_df' column wise.

            logging.info(f'save preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)