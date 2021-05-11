import datetime
import math

import numpy as np
import pandas as pd
from bson import ObjectId
from singleton_decorator import singleton
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class LabelBuilder(BaseEstimator, TransformerMixin):
    """
        Age of the URL (Label)
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def __transform(self, x, y=None):
        result = x
#         result.loc[:, 'label'] = result.last_available_timestamp \
#             .apply(self._convert_timestamp_to_coef) \
#             .fillna(self._extract_year(result.id.apply(ObjectId))) \
#             .astype(int)
        result.loc[:, 'last_appear'] = result.last_available_timestamp \
            .apply(self._convert_timestamp_to_coef) \
            .fillna(self._extract_year(result.id.apply(ObjectId))) \
            .astype(int)
        return result
        
    def transform(self, x, y=None):
        result = x
        first_appear = result.first_appear.fillna(self._extract_year(result.id.apply(ObjectId)))
        last_appear = result.last_available_timestamp \
            .apply(self._convert_timestamp_to_coef) \
            .fillna(self._extract_year(result.id.apply(ObjectId))) \
            .astype(int)
        result.loc[:, 'label'] = last_appear - first_appear
        result = result[result.label.apply(lambda _x: not math.isnan(_x))]
        result = result[result.label >= 0]
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


class ColumnRenamer(BaseEstimator, TransformerMixin):
    """
        Protocol Type Conversion
    """
    def __init__(self, mapping):
        self._mapping = mapping

    @property
    def mapping(self):
        return self._mapping

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        self._mapping = {key: value for key, value in self._mapping.items() if key in result.columns}
        result = result.rename(columns=self._mapping)
        return result


class FeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self._removed_features = None
        self._features = features

    @property
    def removed_features(self):
        return self._removed_features

    def fit(self,x,y=None):
        return self

    def transform(self,x,y=None):
        result = x
        self._removed_features = [col_name for col_name in self._features if col_name in result.columns]
        result = result.drop(self._removed_features, axis=1)
        return result


class FeaturePicker(BaseEstimator, TransformerMixin):
    """
    Remove redundant features
    """
    def __init__(self, features):
        self._picked_features = None
        self._features = features

    @property
    def picked_features(self):
        return self._picked_features

    def fit(self, x, y=None):
        return self

    def transform(self,x,y=None):
        result = x
        self._picked_features = [col_name for col_name in self._features if col_name in result.columns]
        result = result[self._picked_features]
        return result


class CustomizedStandardizer(BaseEstimator, TransformerMixin):
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
        for col_name in result.drop('label', axis=1).columns:
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
        self._columns = [*binary_columns, *numeric_columns, *other_columns, 'label']

#         result.loc[:, 'label'] = x.label-1970
        return result[self._columns]


class FeatureValueMapper(BaseEstimator, TransformerMixin):
    """
    Convert binary features into numeric variables
    """
    def __init__(self, column_name, mapping):
        self._column_name = column_name
        self._mapping = mapping

    @property
    def column_name(self):
        return self._column_name

    @property
    def mapping(self):
        return self._mapping

    def fit(self, x, y=None):
        result = x
        result.loc[:, self._column_name] = result[self._column_name].map(self._mapping)
        return self

    def transform(self, x, y=None):
        result = x
        return result


class LogarithmTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self._columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result.loc[:, self._columns] = (result[self._columns] + 0.00000000001).applymap(math.log)

        return result


class NanToZeroConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self._columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        self._columns = [col_name for col_name in self._columns if col_name in result.columns]
        for col_name in self._columns:
            result.loc[:, col_name] = result[col_name].fillna(0)
        return result
