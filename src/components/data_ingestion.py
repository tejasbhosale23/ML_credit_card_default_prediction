import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_date_ingestion(self):
        logging.info('entered the data ingestion method or component')

        try:
            df = pd.read_csv('UCI_Credit_Card.csv')
            
            df['EDUCATION'] = df['EDUCATION'].replace([0,5,6], 4)

            df['MARRIAGE'] = df['MARRIAGE'].apply(lambda x: 3 if x==0 else x)

            df.drop('ID', axis=1, inplace=True)

            Q1 = df['AGE'].quantile(0.25)
            Q3 = df['AGE'].quantile(0.75)
            IQR = Q3 - Q1
            min = Q1 - 1.5*IQR
            max = Q3 + 1.5*IQR
            df = df[(df['AGE'] >= min) & (df['AGE'] <= max)]

            Q1 = df['LIMIT_BAL'].quantile(0.25)
            Q3 = df['LIMIT_BAL'].quantile(0.75)
            IQR = Q3 - Q1
            min = Q1 - 1.5*IQR
            max = Q3 + 1.5*IQR
            df = df[(df['LIMIT_BAL'] >= min) & (df['LIMIT_BAL'] <= max)]


            df['SEX'].replace({1: 'MALE', 2:'FEMALE'}, inplace= True)
            df['EDUCATION'].replace({1: 'graduate school', 2: 'university', 3: 'high school', 4:'others'}, inplace=True)
            df['MARRIAGE'].replace({1: 'married', 2: 'single', 3: 'others'}, inplace= True)

            df.rename(columns={'default.payment.next.month': 'default'}, inplace=True)

            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('train test split initiated')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of the data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_date_ingestion()