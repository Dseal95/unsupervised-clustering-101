"""utils module contains useful functions to be used when running the UnsupervisedML.ipynb."""

import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

np.random.seed(42)


def generate_3d_synthetic_blob_data(
    mus: list, sigmas: list, S: int, add_noise: bool = False
):
    """Generate 3d data using a list of means and stds."""
    # random seed to make synthetic data generation deterministic
    np.random.seed(42)

    blob1 = np.random.normal(mus[0], sigmas[0], size=(S, 3))
    blob2 = np.random.normal(mus[1], sigmas[1], size=(S, 3))
    blob3 = np.random.normal(mus[2], sigmas[2], size=(S, 3))

    data = [blob1, blob2, blob3]

    if add_noise:
        noise = np.random.uniform((-1 * max(mus)) - 2, max(mus) + 2, size=(S // 10, 3))
        data.append(noise)

    coordinates = np.concatenate(data)
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    return x, y, z


def generate_2d_synthetic_blob_data(
    mus: list, sigmas: list, S: int, add_noise: bool = False
):
    """Generate 3d data using a list of means and stds."""
    # random seed to make synthetic data generation deterministic
    np.random.seed(42)

    blob1 = np.random.normal(mus[0], sigmas[0], size=(S, 2))
    blob2 = np.random.normal(mus[1], sigmas[1], size=(S, 2))

    data = [blob1, blob2]

    if add_noise:
        noise = np.random.uniform((-1 * max(mus)) - 2, max(mus) + 2, size=(S // 10, 2))
        data.append(noise)

    coordinates = np.concatenate(data)

    x, y = coordinates[:, 0], coordinates[:, 1]

    return x, y


def join_data(X: pd.DataFrame, model):
    """Combine the clustered data with the cluster labels into a pandas df."""
    cluster_labels = pd.DataFrame(
        data=model.labels_, index=X.index, columns=["cluster"]
    )

    data = X.merge(cluster_labels, how="inner", left_index=True, right_index=True)

    return data


def scaler(df: pd.DataFrame, features: list):
    """Standardise the data.

    Keyword arguments:
    ------------------
    df -- well data
    features -- well features
    """
    # standardise the data using mu and std
    scaled_data = scale(df[features], with_mean=True, with_std=True)
    df_scaled = pd.DataFrame(scaled_data, columns=[features])
    df_scaled.index = df.index
    return df_scaled


def euclidean_dist_2d(c1: list, c2: list):
    """Return the euclidean distance between two 2d coordinates."""
    # error handling
    if not isinstance(c1, list):
        raise ValueError("c1 needs to be a list in form [x1, y1]")

    if not isinstance(c2, list):
        raise ValueError("c2 needs to be a list in form [x2, y2]")

    x1, y1 = c1[0], c1[1]
    x2, y2 = c2[0], c2[1]

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def euclidean_dist_3d(c1: list, c2: list):
    """Return the euclidean distance between two 3d coordinates."""
    # error handling
    if not isinstance(c1, list):
        raise ValueError("c1 needs to be a list in form [x1, y1, z1]")

    if not isinstance(c2, list):
        raise ValueError("c2 needs to be a list in form [x2, y2, z2]")

    x1, y1, z1 = c1[0], c1[1], c1[2]
    x2, y2, z2 = c2[0], c2[1], c2[2]

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
