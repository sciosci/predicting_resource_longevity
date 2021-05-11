import gc
import multiprocessing
import warnings
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


warnings.filterwarnings("ignore")
cpu_cnt = multiprocessing.cpu_count()
allocated_cpu = cpu_cnt
print(f"Allocated {allocated_cpu} CPUs")
gc.collect()


class AnalysisEngineBuilder:

    def __init_(self):
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._param_grid = None
        self._engine = None
        self._train_strategy = None

    def set_X_train(self, X_train):
        self._X_train = X_train
        return self

    def set_y_train(self, y_train):
        self._y_train = y_train
        return self

    def set_X_test(self, X_test):
        self._X_test = X_test
        return self

    def set_y_test(self, y_test):
        self._y_test = y_test
        return self

    def set_param_grid(self, param_grid):
        self._param_grid = param_grid
        return self

    def set_engine(self, engine):
        self._engine = engine
        return self

    def set_train_strategy(self, train_strategy):
        self._train_strategy = train_strategy
        return self

    def build(self):
        strategy = self._train_strategy(
            self._X_train
            , self._y_train
            , self._X_test
            , self._y_test
            , self._param_grid
            , self._engine
        )
        return strategy.execute()

    class TrainStrategy:
        def __init__(self, X_train, y_train, X_test, y_test, param_grid, engine):
            self._X_train = X_train
            self._y_train = y_train
            self._X_test = X_test
            self._y_test = y_test
            self._param_grid = param_grid
            self._engine = engine

        def execute(self):
            raise NotImplementedError()

    class Result:
        def __init__(self, best_result, performance_matrix):
            self._best_result = best_result
            self._performance_matrix = performance_matrix

        @property
        def best_result(self):
            return self._best_result

        @property
        def performance_matrix(self):
            return self._performance_matrix

    class PerformanceResult:
        def __init__(self, model, pred, mse, residual, r_2, rpt, params):
            self._model = model
            self._pred = pred
            self._residual = residual
            self._mse = mse
            self._r_2 = r_2
            self._rpt = rpt
            self._params = params

        @property
        def model(self):
            return self._model

        @property
        def pred(self):
            return self._pred

        @property
        def residual(self):
            return self._residual

        @property
        def mse(self):
            return self._mse

        @property
        def r_2(self):
            return self._r_2

        @property
        def rpt(self):
            return self._rpt

        @property
        def params(self):
            return self._params


class GridSearchStrategy(AnalysisEngineBuilder.TrainStrategy):
    def __init__(self, X_train, y_train, X_test, y_test, param_grid, engine):
        super(GridSearchStrategy, self).__init__(X_train, y_train, X_test, y_test, param_grid, engine)
        self._grid = GridSearchCV(self._engine, self._param_grid, cv=10, scoring='neg_mean_squared_error')

    def execute(self):
        self._grid.fit(self._X_train, self._y_train)

        # Train
        performance_train = self._build_performance_result(self._X_train, self._y_train, 'Training Set')

        # Test
        performance_test = self._build_performance_result(self._X_test, self._y_test, 'Testing Set')
        return performance_train, performance_test

    def _build_performance_result(self, X, y, set_name):
        pred = self._grid.predict(X)
        mse = mean_squared_error(y, pred)
        residual = y - pred
        r_2 = r2_score(y, pred)
        rpt = self._build_static_result(self._grid.best_estimator_, X, y)
        bestResult = GridSearchStrategy._BestPerformanceResult(set_name, self._grid.best_estimator_, pred, mse, residual, r_2, rpt)
        return AnalysisEngineBuilder.Result(bestResult, None)

    def _build_static_result(self, model, X, y):
        pred = model.predict(X)
        params = np.append(model.intercept_, model.coef_)

        newX = np.append(np.ones((len(X), 1)), X, axis=1)
        mse = (sum((y - pred) ** 2)) / (len(newX) - len(newX[0]))

        var_b = mse * (np.linalg.pinv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b

        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

        sd_b = np.round(sd_b, 3)
        ts_b = np.round(ts_b, 3)
        p_values = np.round(p_values, 3)
        params = np.round(params, 4)

        rpt = pd.DataFrame()

        rpt["Coefficients"] = params
        rpt["Standard Errors"] = sd_b
        rpt["t values"] = ts_b
        rpt["Probabilities"] = p_values
        rpt.index = ['Constant', *X.columns.tolist()]

        rpt = rpt.sort_values(by=['Coefficients'], ascending=False)
        return rpt

    class _BestPerformanceResult(AnalysisEngineBuilder.PerformanceResult):
        def __init__(self, set_name, model, pred, mse, residual, r_2, rpt):
            super(GridSearchStrategy._BestPerformanceResult, self).__init__(model, pred, mse, residual, r_2, rpt, None)
            self._set_name = set_name

        def show_performance(self):
            print(f"R^2 on {self._set_name}: {round(self.r_2 * 100, 2)}%")
            display(self.rpt)


class VerboseGridSearchStrategy(GridSearchStrategy):
    def __init__(self, X_train, y_train, X_test, y_test, param_grid, engine):
        super(GridSearchStrategy, self).__init__(X_train, y_train, X_test, y_test, param_grid, engine)
        self._combination_list = pd.DataFrame({'dummy': [1]})
        for key, values in param_grid.items():
            self._combination_list = pd.merge(self._combination_list, pd.DataFrame({key: values, 'dummy': [1] * len(values)}))
        self._combination_list.drop('dummy', axis=1, inplace=True)

    def execute(self):
        # Train and extract scores
        futures = list()
        # Execute models in threads
        with ThreadPoolExecutor(max_workers=allocated_cpu) as executor:
            for combination in self._combination_list.to_dict('records'):
                combination = {key: [value] for key, value in combination.items()}
                future_model = executor.submit(self._train_func(combination))
                futures.append(future_model)
            perf_matrix_train = list()
            perf_matrix_test = list()
            best_train = None
            best_test = None
            best_train_r2 = 0
            best_test_r2 = 0
            for future in futures:
                res = future.result()
                perf_matrix_train.append(res[0])
                perf_matrix_test.append(res[1])
                if res[0].r_2 > best_train_r2:
                    best_train_r2 = res[0].r_2
                    best_train = res[0]
                if res[1].r_2 > best_test_r2:
                    best_test_r2 = res[1].r_2
                    best_test = res[1]

            result_train = AnalysisEngineBuilder.Result(best_train, perf_matrix_train)
            result_test = AnalysisEngineBuilder.Result(best_test, perf_matrix_test)
            return result_train, result_test

    def _train_func(self, params):
        def _train():
            grid = GridSearchCV(self._engine, params, cv=10, scoring='neg_mean_squared_error')
            grid.fit(self._X_train, self._y_train)

            perf_train = self._get_performance(grid.best_estimator_, self._X_train, self._y_train, params)
            perf_test = self._get_performance(grid.best_estimator_, self._X_test, self._y_test, params)

            return perf_train, perf_test

        return _train

    def _get_performance(self, model, X, y, params):
        pred = model.predict(X)
        mse = mean_squared_error(y, pred)
        residual = y - pred
        r_2 = r2_score(y, pred)
        rpt = self._build_static_result(model, X, y)

        return AnalysisEngineBuilder.PerformanceResult(model, pred, mse, residual, r_2, rpt, params)

    # @staticmethod
    # def flat_report():
