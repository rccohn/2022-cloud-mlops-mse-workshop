import json
import matplotlib.pyplot as plt
import mlflow
from sgc_model import SGCNet, mean_acc_and_loss, train_loop
import numpy as np
from pathlib import Path
from data import load_data
from sklearn.metrics import confusion_matrix
from torch.optim import Adam
from torch_geometric.data import Data
from visualize import agg_cm
from sys import argv


def main():
    # read arguments from command line
    ftrain, fval, ftest = argv[1:4]
    k, train_iter, checkpoint_iter = (int(float(x)) for x in argv[4:7])
    lr, decay = (float(x) for x in argv[7:9])
    
    with mlflow.start_run(nested=False):
        mlflow.set_tag('mlflow.runName', 'cgr-SGC')

        # load data
        data_train, data_val, data_test = (load_data(x) for x in (ftrain, fval, ftest))
        for key in ('x', 'y', 'edge_index', 'candidate_mask'):
            print(key, data_train[key].shape, data_train[key].dtype)
        print("running experiment for k={}, lr={}, decay={}".format(k, lr, decay))
        model = SGCNet(k=k, data=data_train)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=decay)
        mlflow.log_params({'learning_rate': lr, 'weight_decay': decay,
                           'loss': 'nll', 'optimizer': 'Adam'})

        sd, train_metrics = train_loop(model, data_train, data_val, data_test, optimizer,
                                      train_iter,
                                       checkpoint_iter,)
        model.load_state_dict(sd)
        mlflow.pytorch.log_model(model, artifact_path='models/SGC')

        # confusion matrices
        yp_train = model.predict(data_train, mask=data_train.candidate_mask)
        yp_val = model.predict(data_val, mask=data_val.candidate_mask)
        yp_test = model.predict(data_test, mask=data_test.candidate_mask)

        y_train = data_train.y[data_train.candidate_mask]
        y_val = data_val.y[data_val.candidate_mask]
        y_test = data_test.y[data_test.candidate_mask]

        cmlist = [confusion_matrix(gt, pred) for gt, pred in zip((y_train, y_val, y_test),
                                                                 (yp_train, yp_val, yp_test))]

        mlflow.log_figure(agg_cm(cmlist), 'figures/confusion_matrix.png')
       


if __name__ == "__main__":
    main()
