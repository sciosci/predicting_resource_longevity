import os

import findspark

findspark.init('/opt/cloudera/parcels/SPARK2-2.3.0.cloudera3-1.cdh5.13.3.p0.458809/lib/spark2/')
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as fn
from singleton_decorator import singleton
from sklearn.model_selection import train_test_split

from Utility import *


@singleton
class DataSource:
    def __init__(self, fract=.04, training_rate=.7, seed=seed):
#     def __init__(self, fract=.003, training_rate=.7, seed=seed):
        self._spark = None
        self._sc = None
        self._spark_sql = None
        self._raw_data = None
        self._train_data = None
        self._test_data = None

        self._label_name = 'label'
        self._cache_dataset_file_name = 'raw_data_01.json'
        self.re_initialize(fract, training_rate, seed)
        print('Initialized')

    def load_dataset(self, path, name):
        return self._spark.read.parquet(path).registerTempTable(name)

    def re_initialize(self, fract, training_rate, seed):
        if os.path.isfile(self._cache_dataset_file_name):
            self._raw_data = pd.read_json(self._cache_dataset_file_name, orient='index')
        else:
            self._spark = SparkSession.builder. \
                config('spark.app.name', 'elastic_net_reg'). \
                config('spark.dynamicAllocation.enabled', 'true'). \
                config('spark.dynamicAllocation.maxExecutors', '50'). \
                config('spark.dynamicAllocation.executorIdleTimeout', '30s'). \
                config('spark.driver.maxResultSize', '8g'). \
                config('spark.driver.memory', '50g'). \
                config('spark.executor.memory', '10g'). \
                config('spark.task.maxFailures', '3'). \
                config('spark.yarn.am.memory', '50g'). \
                config('spark.yarn.max.executor.failures', '3'). \
                config('spark.kryoserializer.buffer.max', '1024m'). \
                config('spark.yarn.executor.memoryOverhead', '50g'). \
                getOrCreate()
            self._sc = self._spark.sparkContext
            self._spark_sql = SQLContext(self._sc)
            print(self._spark.version)

            self.load_dataset('/user/jjian03/WebResourceQuality.parquet', 'web_resource_quality')
            self.load_dataset('/user/jjian03/WebResourceQuality_pmid.parquet', 'web_resource_quality_pmid')
            self.load_dataset('/datasets/MAG_20200403/MAG_Azure_Parquet/mag_parquet/Papers.parquet', 'Paper')
            self.load_dataset('/user/lliang06/icon/MAG_publication_features.parquet', 'mag')

            self._raw_data = self._spark_sql.sql('''
                    SELECT wr.id
                        , wr.url
                        , wr.actual_scrape_url
                        , wr.first_appear
                        , wr.first_available_timestamp
                        , wr.last_available_timestamp
                        , wr.header
                        , wr.html_text
                        , wr.comment
                        , wr.from_waybackmachine
                        , wr.http_status_code
                        , wr.original_check_failure
                        , wr.original_check_error_log
                        , wr.terminate_reason
                        , wr.terminate_reason_error_log
    
                        , m.paperId
                        , m.total_num_of_paper_citing
                        , m.total_num_of_author_citing
                        , m.total_num_of_affiliation_citing
                        , m.total_num_of_journal_citing
                        , m.total_num_of_author_self_citation
                        , m.total_num_of_affiliation_self_citation
                        , m.total_num_of_journal_self_citation
                        , m.avg_year
                        , m.min_year
                        , m.max_year
                        , m.median
                        , m.num_of_author
                        , m.num_of_author_citing
                        , m.num_of_affiliation_citing
                        , m.num_of_journal_citing
                        , m.avg_hindex
                        , m.first_author_hindex
                        , m.last_author_hindex
                        , m.avg_mid_author_hindex
                        , m.paper_unique_affiliation
    
                    FROM web_resource_quality wr
                    JOIN web_resource_quality_pmid wr_doi ON wr.id = wr_doi.id
                    JOIN Paper p ON wr_doi.doi = p.doi
                    JOIN mag m ON p.paperId = m.paperId
                    WHERE wr.label IS NOT NULL
                    AND wr.label IN ("0", "1")
                    AND isNaN(wr.label) = false
                    AND wr.first_appear IS NOT NULL
                    AND isNaN(wr.first_appear) = false
                    AND lower(wr.url) NOT LIKE "%doi.org%"
                ''') \
                .orderBy(fn.rand(seed=seed)) \
                .sample(False, fract, seed) \
                .toPandas()
            self._raw_data.to_json(self._cache_dataset_file_name, orient='index')

        print(f'Sample Size - raw_data: {len(self._raw_data)}')

        self._train_data, self._test_data = train_test_split(self._raw_data, test_size=1-training_rate, random_state=seed)

    @property
    def sparkContext(self):
        return self._sc

    @property
    def sparkSQL(self):
        return self._spark_sql

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data
