# encoding utf-8

import torch
from utils.utils import RMSE, MAE, SMAPE
from utils.utils import Un_Z_Score
import numpy as np
import os


def Cal_eval_index(epoch, pred, loss_meathod, val_target, time_slice, mean, std):
    val_index = {}
    val_index['MAE'] = []
    val_index['RMSE'] = []
    val_index['sMAPE'] = []
    val_loss = []

    if torch.cuda.is_available():
        mean = torch.tensor(mean).cuda()
        std = torch.tensor(std).cuda()

    for item in time_slice:
        pred_index = pred[:, :, item - 1]
        val_target_index = val_target[:, :, item - 1]
        pred_index, val_target_index = Un_Z_Score(pred_index, mean, std), Un_Z_Score(val_target_index, mean, std)

        loss = loss_meathod(pred_index, val_target_index)
        val_loss.append(loss)

        filePath = "./results/"
        if not os.path.exists(filePath):
            os.makedirs(filePath)

        if not os.path.exists(filePath):
            os.makedirs(filePath)
        if ((epoch+1) % 50 == 0) & (epoch != 0) & (epoch > 200):
            np.savetxt(filePath + "/pred_" + str(epoch) + ".csv", pred_index.cpu(), delimiter=',')
            np.savetxt(filePath + "/true_" + str(epoch) + ".csv", val_target_index.cpu(), delimiter=',')

        mae = MAE(val_target_index, pred_index)
        val_index['MAE'].append(mae)

        rmse = RMSE(val_target_index, pred_index)
        val_index['RMSE'].append(rmse)

        smape = SMAPE(val_target_index, pred_index)
        val_index['sMAPE'].append(smape)

    return val_loss, val_index


def Evaluate(epoch, model, loss_meathod, NATree, time_slice, data_set):
    model.eval()
    eval_input = data_set['eval_input']
    eval_target = data_set['eval_target']

    if torch.cuda.is_available():
        eval_input = eval_input.cuda()
        val_target = eval_target.cuda()
    pred = model(NATree, eval_input)

    eval_loss, eval_index = Cal_eval_index(epoch, pred, loss_meathod, val_target, time_slice, data_set['X_mean'], data_set['X_std'])
    return eval_loss, eval_index
