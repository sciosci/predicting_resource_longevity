import math

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    @staticmethod
    def plot_residual(df):
        titles = df.title.unique()
        fig = go.Figure()

        buttons = list()
        buttons.append(
            dict(label='Residual Scatter Plot',
                 method="update",
                 args=[{"visible": [True for title in titles]},
                       {"title": '',
                        "annotations": []}])
        )
        visible = [False for title in titles]
        for idx, title in enumerate(titles):
            sub_df = df[df.title == title]
            fig.add_trace(go.Scatter(
                x=sub_df.prediction,
                y=sub_df.residual,
                mode='markers',
                name=title
            ))

            _v = visible.copy()
            _v[idx] = True
            buttons.append(
                dict(label=title,
                     method="update",
                     args=[{"visible": _v},
                           {"title": '',
                            "annotations": []}]),
            )

        x_span = (df.prediction.max() - df.prediction.min()) / 20
        y_span = (df.residual.max() - df.residual.min()) / 20

        fig.add_shape(
            # Line Horizontal
            type="line",
            x0=df.prediction.min() - x_span,
            y0=0,
            x1=df.prediction.max() + x_span,
            y1=0,
            line=dict(
                color="Red",
                width=4,
                dash="dashdot",
            ),
        )

        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    pad={"l": 2, "t": 10},
                    showactive=False,
                    x=0,
                    xanchor="left",
                    y=1.2,
                    yanchor="top"
                ),
            ])
        fig.update_layout(showlegend=False)
        fig.update_xaxes(range=[df.prediction.min() - x_span, df.prediction.max() + x_span])
        fig.update_yaxes(range=[df.residual.min() - y_span, df.residual.max() + y_span])

        # Draw the trending area
        _01_pred = df.prediction.quantile(.01)
        _35_pred = df.prediction.quantile(.35)
        _75_pred = df.prediction.quantile(.75)
        _90_pred = df.prediction.quantile(.90)
        left_upper = (df.prediction.min(), df.residual[df.prediction < _01_pred].quantile(.90))
        left_lower = (df.prediction.min(), df.residual[df.prediction < _01_pred].quantile(.10))

        mid_upper = (df.prediction.mode().iloc[0],
                     df.residual[(df.prediction > _35_pred) & (df.prediction < _75_pred)].quantile(.90))
        mid_lower = (df.prediction.mode().iloc[0],
                     df.residual[(df.prediction > _35_pred) & (df.prediction < _75_pred)].quantile(.10))

        right_upper = (df.prediction.max(), df.residual[df.prediction > _90_pred].quantile(.90))
        right_lower = (df.prediction.max(), df.residual[df.prediction > _90_pred].quantile(.10))

        fig.update_layout(
            shapes=[
                dict(
                    type="path",
                    path=f'''
                        M {left_upper[0]},{left_upper[1]} 
                        L{mid_upper[0]},{mid_upper[1]} 
                        L{right_upper[0]},{right_upper[1]} 
                        L{right_lower[0]},{right_lower[1]} 
                        L{mid_lower[0]},{mid_lower[1]} 
                        L{left_lower[0]},{left_lower[1]} 
                        Z''',
                    fillcolor="PaleTurquoise",
                    line_color="LightSeaGreen",
                    opacity=0.5,
                ),
            ]
        )

        fig.show()

    @staticmethod
    def plot_scatter_distribution(ivs, dv, title):
        n_col = 3
        n_row = math.ceil(len(ivs.columns) / 3)
        fig = make_subplots(rows=n_row, cols=n_col, subplot_titles=ivs.columns)
        for idx, col_name in enumerate(ivs.columns):
            fig.add_trace(
                go.Scatter(
                    x=ivs[col_name],
                    y=dv,
                    mode='markers',
                    name=col_name
                ), row=int(idx / 3) + 1, col=idx % 3 + 1)

        fig.update_layout(title_text=title)
        fig.show()

    @staticmethod
    def plot_loss_trend(data, x_name, y_name, facet, title):
        fig = go.Figure()
        legends = data[facet].unique()
        for legend in legends:
            sub_data = data[data[facet] == legend]
            sub_data = sub_data.groupby(x_name).agg(['mean'])[y_name]
            sub_data[x_name] = sub_data.index
            sub_data = sub_data.rename(columns={"mean": y_name})

            fig.add_trace(go.Scatter(x=sub_data[x_name], y=sub_data[y_name],
                                     mode='lines',
                                     name=legend))

        fig.update_layout(title_text=title)
        fig.update_layout(showlegend=True)
        fig.show()

    @staticmethod
    def plot_feature_importance(reg_coef, col_names, title):
        reg_coef = pd.Series(reg_coef, index=col_names)
        reg_coef = reg_coef.sort_values()
        matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
        reg_coef.plot(kind="barh", )
        plt.title(title, fontsize=15)

        return plt

    @staticmethod
    def plot_importance_trending(X_train, feature_importance_matrix, title, offset=3):
        #         return
        feature_importance = feature_importance_matrix.groupby('alpha').agg(['mean'])[[*X_train.columns]]
        feature_importance.columns = X_train.columns.tolist()
        feature_importance['alpha'] = feature_importance.index

        column_names = X_train.columns
        lbds = feature_importance['alpha'].tolist()
        coef_matrix = feature_importance[X_train.columns]
        x_lab = 'Lambda'
        y_lab = 'Weight'
        plt.clf()
        plt.figure(figsize=(15, 10))
        for idx, col_name in enumerate(column_names):
            plt.plot(lbds, coef_matrix.iloc[:, idx], 'o-', linewidth=2, label=col_name)
            c = coef_matrix.iloc[0, idx]
            plt.annotate(col_name, (lbds[offset], coef_matrix.iloc[offset, idx]))

        plt.title(title, fontSize=25)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)

        plt.legend(loc='upper right')
        plt.tight_layout()

        return plt

    @staticmethod
    def plot_distribution(data, title, height=1200, width=800):

        fig = make_subplots(rows=len(data.columns), cols=1,
                            subplot_titles=data.columns)

        for idx, col_name in enumerate(data.columns):
            curve = ff.create_distplot([data[col_name]], [col_name], show_hist=True).data
            fig.add_trace(curve[0], row=idx + 1, col=1)
            fig.add_trace(curve[1], row=idx + 1, col=1)

        fig.update_layout(height=height, width=width, title_text=title)
        return fig
