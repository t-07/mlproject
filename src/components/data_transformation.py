import sys
import os

from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object


@dataclass  #in traditional classes would require __init__ method to define class variables but with this data class decorator we can define class variables as is
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        
        try:
            num_variables=['reading score','writing score']
            cat_variables=[
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course']
            
            num_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy='median')),
                ("Scaler",StandardScaler())]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("One Hot Encoding",OneHotEncoder()),
                    ("Scaler",StandardScaler(with_mean=False))] #with_mean is used so that scaling does not give -ve values, used for sparse datasets(having lot of zeros) i.e. when mean is subtracted from features it should not give negative values
            )


            logging.info(f"Categorical columns: {cat_variables}")
            logging.info(f"Numerical columns: {num_variables}")

            preprocessor=ColumnTransformer(
                [
                    ("num pipeline",num_pipeline,num_variables),
                    ("cat pipeline",cat_pipeline,cat_variables)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_data, test_data):
        
        try:
            train_df=pd.read_csv(train_data)
            test_df=pd.read_csv(test_data)

            logging.info("READ TRAIN AND TEST DATA COMPLETED")

            logging.info("Getting Preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            X_Train=train_df.drop(columns=['math score'], axis=1)
            Y_Train=train_df['math score']

            X_Test=test_df.drop(columns=['math score'], axis=1)
            Y_Test=test_df['math score']
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_input_feature_train_arr=preprocessing_obj.fit_transform(X_Train)
            X_input_feature_test_arr=preprocessing_obj.transform(X_Test)

            train_array=np.c_[X_input_feature_train_arr,np.array(Y_Train)]
            test_array=np.c_[X_input_feature_test_arr,np.array(Y_Test)]

            logging.info(f"Saved preprocessing object.")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
            
        