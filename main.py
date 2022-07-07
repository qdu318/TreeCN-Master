import os
import argparse
import torch
import torch.nn as nn
from utils.data_load import Data_load
from methods.train import Train
from methods.evaluate import Evaluate
import logger
from model.TreeCNs import TreeCNs

torch.cuda.current_device()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--weight_file', type=str, default='./saved_weights/')
parser.add_argument('--timesteps_input', type=int, default=12)
parser.add_argument('--timesteps_output', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=64)
parser.add_argument('--spatial_channels', type=int, default=16)
parser.add_argument('--features', type=int, default=1)
parser.add_argument('--time_slice', type=list, default=[1, 2, 3])

args = parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(7)
    elogger = logger.Logger('run_log')

    NATree, data_set = Data_load(args.timesteps_input, args.timesteps_output)

    N = NATree.shape[0]
    MaxNodeNumber = NATree.shape[2]
    MaxLayerNumber = NATree.shape[1]
    model = TreeCNs(
        num_nodes=N,
        spatial_channels=args.spatial_channels,
        timesteps_output=args.timesteps_output,
        max_node_number=MaxNodeNumber,
    )

    if torch.cuda.is_available():
        model.cuda()
        NATree = torch.from_numpy(NATree).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    L2 = nn.MSELoss()
    for epoch in range(args.epochs):
        print("Train Process")
        train_loss = Train(
            model=model,
            optimizer=optimizer,
            loss_meathod=L2,
            NATree=NATree,
            data_set=data_set,
            batch_size=args.batch_size
        )
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Evalution Process")
            eval_loss, eval_index = Evaluate(
                epoch=epoch,
                model=model,
                loss_meathod=L2,
                NATree=NATree,
                time_slice=args.time_slice,
                data_set=data_set,
            )
        print("---------------------------------------------------------------------------------------------------")
        print("epoch: {}/{}".format(epoch, args.epochs))
        print("Training loss: {}".format(train_loss))
        for i in range(len(args.time_slice)):
            print("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                  .format(args.time_slice[i] * 5, eval_loss[-(len(args.time_slice) - i)],
                          eval_index['MAE'][-(len(args.time_slice) - i)],
                          eval_index['RMSE'][-(len(args.time_slice) - i)],
                          eval_index['sMAPE'][-(len(args.time_slice) - i)]))
            elogger.log("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                        .format(args.time_slice[i] * 5, eval_loss[-(len(args.time_slice) - i)],
                                eval_index['MAE'][-(len(args.time_slice) - i)],
                                eval_index['RMSE'][-(len(args.time_slice) - i)],
                                eval_index['sMAPE'][-(len(args.time_slice) - i)]))
        print("---------------------------------------------------------------------------------------------------")

