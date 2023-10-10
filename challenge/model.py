import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from joblib import load

from typing import Tuple, Union, List

from .utils import (
    threshold_in_minutes,
    top_10_features, 
    get_period_day, 
    is_high_season, 
    get_min_diff
)

class DelayModel:

    def __init__(self):
        self._model = self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )
        if target_column:
            _, _, y_train, _ = train_test_split(features, data[target_column], test_size = 0.33, random_state = 42)
            n_y0 = len(y_train[y_train == 0])
            n_y1 = len(y_train[y_train == 1])
            scale = n_y0/n_y1
            self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
            return (features[top_10_features], data[[target_column]])
        else:
            return features[top_10_features]

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        x_train, _, y_train, _ = train_test_split(features, target, test_size = 0.33, random_state = 42)
        self._model.fit(x_train, y_train)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        return self._model.predict(features).tolist()
    
    def load(self):
        self._model = load('./challenge/model.joblib')