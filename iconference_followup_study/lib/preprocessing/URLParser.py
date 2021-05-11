import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from lib.preprocessing.RegularParser import FeatureRemover, FeaturePicker

pd.options.mode.chained_assignment = None

from feature_engine import categorical_encoders
from itertools import compress
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import feature_selection


class URLLengthCounter(BaseEstimator, TransformerMixin):
    """
    Length of the url hierarchy
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result.loc[:, 'url_length'] = result['url'].apply(self._get_length)
        return result

    def _get_length(self, url):
        return len(url)


class URLParser(BaseEstimator, TransformerMixin):
    """
    url parser
    """
    def __init__(self):
        self._lambdas = dict()

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result.loc[:, 'url_parse_obj'] = result['url'].apply(urlparse)

        result.loc[:, 'scheme'] = result.url_parse_obj.apply(lambda _x: _x.scheme)
        result.loc[:, 'netloc'] = result.url_parse_obj.apply(lambda _x: _x.netloc)
        result.loc[:, 'path'] = result.url_parse_obj.apply(lambda _x: _x.path)
        result.loc[:, 'params'] = result.url_parse_obj.apply(lambda _x: _x.query) \
            .apply(lambda _x: None if '' == _x.strip() else _x)

        result = result.drop(['url_parse_obj'], axis=1)
        return result

    def register_new_column(self, col_name, lbd):
        self._lambdas[col_name] = lbd
        return self


class URLDepthCounter(BaseEstimator, TransformerMixin):
    """
    Depth of the url hierarchy
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result.loc[:, 'url_depth'] = result['path'].apply(self._get_depth)
        return result

    def _get_depth(self, path):
        last_idx = path.rindex('/')
        if last_idx + 1 < len(path):
            last_idx = len(path)
        return path[:last_idx].count('/')


class HasWWWConverter(BaseEstimator, TransformerMixin):
    """
    Has WWW subdomain
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result.loc[:, 'has_www'] = result['netloc'].apply(self._has_www)
        return result

    def _has_www(self, domain):
        return int(domain.startswith('www.'))


class SubdomainLevelCounter(BaseEstimator, TransformerMixin):
    """
    Level of the Subdomain
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result.loc[:, 'subdomain_level'] = result['netloc'].apply(self._get_level)
        return result

    def _get_level(self, domain):
        return domain.count('.')


class RequestParameterCounter(BaseEstimator, TransformerMixin):
    """
    Number of HTTP-Get parameters
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result['params'] = result['params'].replace(np.nan, '', regex=True)
        result.loc[:, 'param_cnt'] = result['params'].apply(self._count_param)
        return result

    def _count_param(self, params):
        if params is '':
            return 0
        return params.count('&') + 1


class DomainSuffixBuilder(BaseEstimator, TransformerMixin):
    """
    Domain Suffix
    """
    def __init__(self):
        self._suffix_dict = None

    def fit(self, x, y=None):
        new_features = self.build_suffix_port_feature(x)
        new_features = new_features.dropna()
        encoder = categorical_encoders.CountFrequencyCategoricalEncoder(
            encoding_method='frequency',
            variables=['suffix'])
        encoder.fit(new_features)
        self._suffix_dict = encoder.encoder_dict_['suffix']
        return self

    def transform(self, x, y=None):
        result = x
        new_features = self.build_suffix_port_feature(x)
        for col_name in new_features.columns:
            result.loc[:, col_name] = new_features[col_name]
        result.loc[:, 'suffix'] = result.suffix.apply(lambda v: self._suffix_dict[v] if v in self._suffix_dict else 0)

        result = result.dropna(subset=['is_port_access', 'suffix', 'suffix_idx'])
        return result

    def build_suffix_port_feature(self, x):
        result = x
        # Remove incorrect urls
        #         result = result[result['netloc'].apply(lambda val: '.' in val)]
        # Build features
        suffix = result.netloc.apply(DomainSuffixBuilder._get_url_suffix)
        is_port_access = suffix.apply(DomainSuffixBuilder._is_port_access)
        suffix_idx = suffix.apply(DomainSuffixBuilder._clean_url_suffix)

        return pd.DataFrame({'suffix': suffix, 'suffix_idx': suffix_idx, 'is_port_access': is_port_access, })

    @property
    def suffix_dict(self):
        return self._suffix_dict

    @staticmethod
    def _get_url_suffix(url):
        if not '.' in url:
            return None
        last_idx = url.rindex('.')
        return url[last_idx + 1:]

    @staticmethod
    def _clean_url_suffix(url):
        if None is url:
            return None
        return url.split(':')[0]

    @staticmethod
    def _is_port_access(suffix):
        if None is suffix:
            return None
        return int(len([token for token in suffix.split(':') if token.strip() != '']) > 1)


class IncorrectDomainUrlCleaner(BaseEstimator, TransformerMixin):
    """
    Remove the Incorrect Domains
    TLD ranges from 2 to 63

    Ref: https://en.wikipedia.org/wiki/Domain_Name_System#cite_ref-rfc1034_1-2
    """
    def __init__(self):
        # TLD ranges from 2 to 63
        self._regex = re.compile(r'^[a-zA-Z]{2,63}$', re.I)

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result.loc[:, 'is_correct'] = result.suffix_idx.apply(self._is_correct)
        result = result[result.is_correct]
        result = result.drop('is_correct', axis=1)
        return result

    def _is_correct(self, domain_suffix):
        return True if self._regex.match(domain_suffix) else False


class BinaryNAEncoder(BaseEstimator, TransformerMixin):
    """
    Has content-type
    """
    def __init__(self, columns):
        self._columns = columns

    @property
    def columns(self):
        return self._columns

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        self._columns = [col_name for col_name in self._columns if col_name in result.columns]
        for col_name in self._columns:
            result.loc[:, f'has_{col_name}'] = result[col_name] \
                .apply(lambda x: x not in [np.nan, None]) \
                .map({True: 1, False: 0})

        return result


class LowVarianceRemover(BaseEstimator, TransformerMixin):
    """
    Miscellaneous Clean Up
        Standardize variance
        Convert Categorical Feature into Frequency Based Numberical Index
        Remove low variance features
    """
    def __init__(self, threshold):
        self._p = threshold
        self._bi_vt = feature_selection.VarianceThreshold(threshold=threshold*(1-threshold))
        self._regular_vt = feature_selection.VarianceThreshold(threshold=threshold)
        self._dropped_columns = list()

    @property
    def threshold(self):
        return self._threshold

    @property
    def dropped_columns(self):
        return self._dropped_columns

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x

        df_unique = pd.DataFrame()
        for col_name in result.columns:
            if 'label' != col_name:
                df_unique[col_name] = [len(result[col_name].unique())]

        df_unique.index = ['unique count']
        df_unique = df_unique.T.squeeze()

        bi_columns = df_unique[df_unique == 2].index.tolist()
        regular_columns = df_unique[df_unique != 2].index.tolist()

        if len(bi_columns) > 0:
            self._bi_vt.fit(result[bi_columns])
            bi_mask = self._bi_vt.variances_ < self._p * (1 - self._p)
            self._dropped_columns = self._dropped_columns + list(compress(bi_columns, bi_mask))
        if len(regular_columns) > 0:
            self._regular_vt.fit(result[regular_columns])
            regular_mask = self._regular_vt.variances_ < self._p
            self._dropped_columns = self._dropped_columns + list(compress(regular_columns, regular_mask))

        if len(self._dropped_columns) > 0:
            remover = FeatureRemover(self._dropped_columns)
            result = remover.transform(result)
        return result


class DummySuffixDescritizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        dummies = pd.get_dummies(result.suffix_idx)
        dummies = FeaturePicker(['int', 'org', 'gov', 'in', 'eu', 'cn', 'kr', 'en']).fit_transform(dummies)
        result = result.drop('suffix_idx', axis=1).join(dummies, how='inner')

        return result