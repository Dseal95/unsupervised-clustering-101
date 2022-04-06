"""A plotting module that contains all of the plotting functions used in the NB."""

import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt


def plot_null_heat_map(df: pd.DataFrame):
    """Plot a null heat map highlighting the null values in a pandas df."""
    # NaN Heatmap
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(~df.isnull(), cbar=False, ax=ax)
    plt.show()


def build_plot_layout(
    title: str,
    x_axis_title: str,
    y_axis_title: str,
    h: int = 400,
    w: int = 400,
    background_colour: str = "rgb(255, 255, 255)",
):
    """"""
    # plot layout
    layout = go.Layout(
        title=title,
        plot_bgcolor=background_colour,
        height=h,
        width=w,
        yaxis=dict(
            title=y_axis_title,
            gridcolor="lightgrey",
            showgrid=True,
            showline=True,
            linecolor="black",
            linewidth=2,
            mirror=True,
            ticks="outside",
        ),
        xaxis=dict(
            title=x_axis_title,
            gridcolor="lightgrey",
            showgrid=True,
            showline=True,
            linecolor="black",
            linewidth=2,
            mirror=True,
            ticks="outside",
        ),
    )

    return layout


def plot_3d_scatter(x, y, z, layout: dict, labels=None, h=600, w=600):
    """Plot a 3d scatter plot of x, y and z."""
    if labels is not None:
        label = [i for i in labels]
    else:
        label = "black"

    raw_data = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        showlegend=False,
        opacity=0.5,
        marker=dict(color=label, size=2),
        hovertemplate="<b>value:</b><br>"
        + "x: %{x}<br>"
        + "y: %{y}<br>"
        + "z: %{z}<br>",
    )

    fig = go.Figure([raw_data])
    fig.update_scenes(layout)
    fig.update_layout(width=w, height=h)

    return fig


def plot_2d_scatter(x, y, layout: dict, labels=None):
    """Plot a 2d scatter plot of x and y."""
    if labels is not None:
        label = [i for i in labels]
    else:
        label = "black"

    data = go.Scattergl(
        x=x,
        y=y,
        mode="markers",
        showlegend=False,
        opacity=0.5,
        marker=dict(color=label, size=2),
        hovertemplate="<b>value:</b><br>" + "x: %{x}<br>" + "y: %{y}<br>",
    )

    return go.Figure(data, layout)


def plot_2d_clustered_scatter(df, layout):
    """Plot a 2d scatter plot of x and y with cluster labels. Assumes order  [x, y, cluster_label]"""
    data = go.Scattergl(
        x=df.iloc[:, 0],
        y=df.iloc[:, 1],
        mode="markers",
        showlegend=False,
        opacity=0.5,
        marker=dict(size=2, color=df.iloc[:, 2], colorscale="thermal"),
        hovertemplate="<b>%{hovertext} </b><br><br>" + "x: %{x}<br>" + "y: %{y}<br>",
        hovertext=[f"Cluster = {i}" for i in list(df.iloc[:, 2])],
    )

    return go.Figure(data, layout)


def plot_3d_clustered_scatter(df, layout, h=600, w=600):
    """Plot a 3d scatter plot of x, y and z with cluster labels. Assumes order  [x, y, z, cluster_label]"""
    raw_data = go.Scatter3d(
        x=df.iloc[:, 0],
        y=df.iloc[:, 1],
        z=df.iloc[:, 2],
        mode="markers",
        showlegend=False,
        opacity=0.5,
        marker=dict(size=2, color=df.iloc[:, 3], colorscale="thermal"),
        hovertemplate="<b>%{hovertext} </b><br><br>"
        + "x: %{x}<br>"
        + "y: %{y}<br>"
        + "z: %{z}<br>",
        hovertext=[f"Cluster = {i}" for i in list(df.iloc[:, 3])],
    )

    fig = go.Figure([raw_data])
    fig.update_scenes(layout)
    fig.update_layout(width=w, height=h)

    return fig


def generate_elbow_plots(df, layout: dict, model_type: str):
    """Generate elbow plots for either kmeans or DBSCAN based on silhouette and CH scores."""
    if model_type.lower() == "kmeans":
        col = "n_clusters"
    elif model_type.lower() == "dbscan":
        col = "eps"
    else:
        raise ValueError('Please enter either "kmeans" or "dbscan"')

    layout.update(xaxis=dict(dtick=1))
    layout.update({"height": 600, "width": 600})

    # silhouette plot
    S_data = go.Scatter(
        x=df[col],
        y=df["silhouette"],
        marker=dict(size=8),
        mode="markers+lines+text",
        text=[i for i in df[col].to_list()],
        textposition="top center",
    )

    # CH plot
    CH_data = go.Scatter(
        x=df[col],
        y=df["calinski_harabasz"],
        marker=dict(size=8),
        mode="markers+lines+text",
        text=[i for i in df[col].to_list()],
        textposition="top center",
    )

    return [
        go.Figure(data=S_data, layout=layout),
        go.Figure(data=CH_data, layout=layout),
    ]
