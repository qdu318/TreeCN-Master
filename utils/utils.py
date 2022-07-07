import numpy as np
import torch


def Z_Score(matrix):
    mean, std = np.mean(matrix), np.std(matrix)
    return (matrix - mean) / (std+0.001), mean, std


def Un_Z_Score(matrix, mean, std):
    return (matrix * std) + mean


def get_normalized_adj(W_nodes):
    W_nodes = W_nodes + np.diag(np.ones(W_nodes.shape[0], dtype=np.float32))
    D = np.array(np.sum(W_nodes, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    W_nodes = np.multiply(np.multiply(diag.reshape((-1, 1)), W_nodes),
                         diag.reshape((1, -1)))
    return W_nodes


def generate_asist_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features))


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))


def RMSE(v, v_):
    return torch.sqrt(torch.mean((v_ - v) ** 2))


def MAE(v, v_):
    return torch.mean(torch.abs(v_ - v))


def SMAPE(v, v_):
    """
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    """
    return torch.mean(torch.abs((v_ - v) / ((torch.abs(v) + torch.abs(v_)) / 2 + 1e-5)))

