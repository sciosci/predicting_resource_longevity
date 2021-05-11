from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class SourceCodeByteCounter(BaseEstimator, TransformerMixin):
    """
    Code length(kb)
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result['code_size'] = result.html_text \
            .replace(np.nan, '', regex=True) \
            .astype(str) \
            .apply(len)

        return result


class HTML5Justifier(BaseEstimator, TransformerMixin):
    """
    is HTML5
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result['is_html5'] = result.html_text \
            .replace(np.nan, '', regex=True) \
            .apply(lambda _x: int(
            _x.replace('\n', '').replace('\r', '').strip().lower().startswith('<!doctype html>') if _x else False))

        return result


class BeautifulSoupParserBuilder:
    """
    Text Mining
    """
    class _BeautifulSoupParser(BaseEstimator, TransformerMixin):
        def __init__(self, _lambda_pair):
            self._lambda_pair = _lambda_pair

        def fit(self, x, y=None):
            return self

        def transform(self, x, y=None):
            result = x
            soup_handlers = result.html_text \
                .replace(np.nan, '', regex=True) \
                .apply(lambda html_doc: BeautifulSoupParserBuilder._safe_create_parser(html_doc))

            for col_name, func in self._lambda_pair.items():
                result[col_name] = soup_handlers.apply(func)

            return result

    @staticmethod
    def _safe_create_parser(html_doc):
        try:
            return BeautifulSoup(html_doc, 'html.parser')
        except:
            return BeautifulSoup('', 'html.parser')

    def __init__(self):
        self._lambda_pair = dict()

    def add_lambda(self, column_name, lbd):
        self._lambda_pair[column_name] = lbd
        return self

    def build(self):
        return BeautifulSoupParserBuilder._BeautifulSoupParser(self._lambda_pair)


def get_title_length(soup):
    """
    Title Length
    :param soup:
    :return:
    """
    title = soup.title.string if soup.title else ''
    if not title:
        title = ''
    return len(title)


def count_internal_js_lib(soup):
    """
    No of internal JS files
    :param soup:
    :return:
    """
    sources = soup.findAll('script', {"src": True})
    return len([0 for source in sources if not source['src'].startswith('http')])


def count_external_js_lib(soup):
    """
    No of external JS files
    :param soup:
    :return:
    """
    sources = soup.findAll('script', {"src": True})
    return len([0 for source in sources if source['src'].startswith('http')])


def get_charset(soup):
    """
    Charset
    :param soup:
    :return:
    """
    sources = soup.findAll('meta', {"charset": True})
    if 0 == len(sources):
        return ''
    return sources[0]['charset'].lower().replace('\'', '').replace('"', '')


def has_iframe(soup):
    """
    iFrame in Body
    :param soup:
    :return:
    """
    sources = soup.findAll('iframe')
    return int(0 == len(sources))


def count_hyperlink(soup):
    """
    No of hyperlink
    :param soup:
    :return:
    """
    sources = soup.findAll('a')
    return len([1 for source in sources if source.has_attr('href') and source['href'].lower().startswith('http')])


class EmptyHTMLFilter(BaseEstimator, TransformerMixin):
    """
    Drop the records that does not have html code
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x
        result = result.dropna(subset=['html_text'])

        return result


html_parser = BeautifulSoupParserBuilder() \
    .add_lambda('title_length', get_title_length) \
    .add_lambda('internal_js_cnt', count_internal_js_lib) \
    .add_lambda('external_js_cnt', count_external_js_lib) \
    .add_lambda('charset', get_charset) \
    .add_lambda('has_iframe', has_iframe) \
    .add_lambda('hyperlink_cnt', count_hyperlink) \
    .build()
