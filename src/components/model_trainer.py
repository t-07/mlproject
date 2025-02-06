import sys
import os

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models, save_object

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def inititate_model_training(self,train_arr,test_arr):
        try:

            logging.info("SPLIT TRAIN AND TEST INPUT DATA")

            X_train, y_train, X_test, y_test=(
                train_arr[:,:-1], 
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                )
            
            models={
                    "Random Forest":RandomForestRegressor(),
                    "Linear Regression": LinearRegression(),
                    "Ada Boost Regressor": AdaBoostRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Cat Boost":CatBoostRegressor(verbose=False),
                    "XGB Regressor": XGBRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "K Neighbors": KNeighborsRegressor()
                    }
            
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)

            #get the best model score
            best_model_score=max(sorted(model_report.values()))

            logging.info(f"Model report as received is: {model_report}")
            logging.info(f"Best model score is: {best_model_score}")

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square,best_model_name

        except Exception as e:
            raise CustomException(e,sys)
