import datetime
import math

import numpy as np
import pandas as pd
from bson import ObjectId
from singleton_decorator import singleton
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class TobitLabelBuilder(BaseEstimator, TransformerMixin):
    """
        Age of the URL (Label)
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result.loc[:, 'label'] = result.last_available_timestamp \
            .apply(self._convert_timestamp_to_coef) \
            .fillna(self._extract_year(result.id.apply(ObjectId))) \
            .astype(int)

        result.loc[:, 'label'] = result.loc[:, 'label'] - 1990
        return result

    def _extract_year(self, ids):
        return ids.apply(lambda x: x.generation_time.year)

    def _convert_timestamp_to_coef(self, ts):
        if None is ts or np.nan is ts or math.isnan(ts):
            return ts
        ts_str = str(ts).strip()
        if '' == ts_str:
            return ts

        ts_str = str(int(float(ts_str)))
        ts_obj = datetime.datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        return ts_obj.year


class TobitCustomizedStandardizer(BaseEstimator, TransformerMixin):
    """
    Add Sklearn Build-in Function
    """
    def __init__(self, norm='l2'):
        self._pipe = Pipeline([
            ('standard_scaler', preprocessing.StandardScaler()),

        ])
        self._columns = None

    @property
    def columns(self):
        return self._columns

    def fit(self,x,y=None):
        return self

    def transform(self,x,y=None):
        result = x

        df_unique = pd.DataFrame()
        for col_name in result.drop(['label', 'url'], axis=1).columns:
            df_unique[col_name] = [len(result[col_name].unique())]

        df_unique.index = ['unique count']
        df_unique = df_unique.T.squeeze()

        binary_columns = df_unique[df_unique < 3].index.tolist()
        numeric_columns = x.drop([*binary_columns, 'label'], axis=1).select_dtypes(include=np.number).columns.tolist()
        other_columns = x.drop([*binary_columns, *numeric_columns, 'label'], axis=1).columns.tolist()
        label = x.label.tolist()
        label = np.array([label]).T

        result = label
        if len(binary_columns) > 0:
            result = np.append(result, x[binary_columns], axis=1)
        if len(numeric_columns) > 0:
            numeric_result = self._pipe.fit_transform(x[numeric_columns])
            result = np.append(result, numeric_result, axis=1)
        if len(other_columns) > 0:
            result = np.append(result, x[other_columns], axis=1)

        result = pd.DataFrame(result, columns= ['label', *binary_columns, *numeric_columns, *other_columns])
        result.loc[:,'scaled_first_appear'] = result.first_appear
        result.loc[:,'first_appear'] = x.first_appear
        self._columns = [*result.drop(['label', 'url'], axis=1).columns, 'label', 'url']

#         result.loc[:, 'label'] = x.label-1970
        return result[self._columns]