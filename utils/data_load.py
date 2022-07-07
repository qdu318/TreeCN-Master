# encoding utf-8
import numpy as np
import pandas as pd
from utils.utils import Z_Score
from utils.utils import generate_dataset


def Data_load(timesteps_input, timesteps_output):
    X = pd.read_csv("./data_set/SmallScaleAggregation/V_flow_50.csv", header=None).head(8640).to_numpy(np.float32)
    NATree = np.load("./data_set/SmallScaleAggregation/TreeMatrix_50.npy").astype(np.float32)

    # X = pd.read_csv("./data_set/RandomUniformity/V_flow_50.csv", header=None).head(8640).to_numpy(np.float32)
    # NATree = np.load("./data_set/RandomUniformity/TreeMatrix_50.npy").astype(np.float32)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))
    X, X_mean, X_std = Z_Score(X)

    index_1 = int(X.shape[2] * 0.8)
    train_original_data = X[:, :, :index_1]
    val_original_data = X[:, :, index_1:]

    train_input, train_target = generate_dataset(train_original_data,
                                                 num_timesteps_input=timesteps_input,
                                                 num_timesteps_output=timesteps_output)
    evaluate_input, evaluate_target = generate_dataset(val_original_data,
                                                       num_timesteps_input=timesteps_input,
                                                       num_timesteps_output=timesteps_output)
    data_set = {}
    data_set['train_input'], data_set['train_target'], data_set['eval_input'], data_set[
        'eval_target'], data_set['X_mean'], data_set['X_std'], \
        = train_input, train_target, evaluate_input, evaluate_target, X_mean, X_std

    return NATree, data_set

