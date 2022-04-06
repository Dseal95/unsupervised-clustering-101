"""A layout module that contains static layouts for the graphs plotted in the central NB."""

import plotly.graph_objects as go

scene_3d = dict(
    camera=dict(
        center=dict(x=0, y=0, z=0), eye=dict(x=2, y=-1.6, z=0.5), up=dict(x=0, y=0, z=1)
    ),
    dragmode="turntable",
    xaxis=dict(
        showgrid=True,
        backgroundcolor="rgb(255,255,255)",
        gridcolor="lightgrey",
        title=dict(text=f"x", font=dict(family="Rockwell", size=12)),
        ticks="outside",
        rangemode="tozero",
        tickfont=dict(family="Rockwell", size=10),
    ),
    yaxis=dict(
        showgrid=True,
        backgroundcolor="rgb(255,255,255)",
        gridcolor="lightgrey",
        title=dict(text=f"y", font=dict(family="Rockwell", size=12)),
        tickfont=dict(family="Rockwell", size=10),
    ),
    zaxis=dict(
        showgrid=True,
        backgroundcolor="rgb(255,255,255)",
        gridcolor="lightgrey",
        title=dict(text=f"z", font=dict(family="Rockwell", size=12)),
        tickfont=dict(family="Rockwell", size=10),
    ),
    bgcolor="rgb(255,255,255)",
)


layout_2d = go.Layout(
    xaxis_title="x",
    yaxis_title="y",
    yaxis=dict(
        gridcolor="lightgrey",
        showgrid=True,
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
    ),
    xaxis=dict(
        gridcolor="lightgrey",
        showgrid=True,
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
    ),
    plot_bgcolor="rgb(255, 255, 255)",
    height=400,
    width=400,
)
