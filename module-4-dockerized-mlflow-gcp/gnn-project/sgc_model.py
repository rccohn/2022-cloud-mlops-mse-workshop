import mlflow
from mlflow import log_params, set_tags, log_artifact
from os.path import join
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SGConv
from typing import Union
from tqdm import tqdm


class SGCNet(torch.nn.Module):
    def __init__(self, k: int, data: Union[Data, None] = None, num_features: Union[int, None] = None,
                 num_classes: Union[int, None] = None, log: bool = True):
        super().__init__()
        # get number of features (input shape) and number of classes (output shape)
        if data is not None:
            num_features = data.num_features
            num_classes = data.num_classes = int(data.y.max()+1)

        # SGCNet only has one layer, the SGConv layer
        # multiple iterations of message passing handled with K parameter
        # from pyg documentation: cached should only be set to true for transductive learning
        # (ie where the same graph is used and only the labels of some nodes are unknown)
        self.conv = SGConv(in_channels=num_features, out_channels=num_classes, K=k, cached=False)
        self.double()  # forces weight tensors to double type, preventing difficult-to-debug errors later

        # for logging
        self.model_name = "SGCNet-classification-v1"
        self.k = k
        if log:
            self._log()

    def forward(self, data):
        x = self.conv(data.x, data.edge_index)
        return F.log_softmax(x, dim=1)

    # predict needed for mlflow.torch.save_model to include pyfunc
    def predict(self, data, mask=None):
        yp = self.forward(data).argmax(dim=1)
        if mask is None:
            return yp
        else:
            return yp[mask]

    def _log(self):
        set_tags({
            'model': self.model_name
            }
        )

    def save_state(self, fname, artifact_path="model-checkpoints"):
        with TemporaryDirectory() as tmp_dir:
            savepath = join(tmp_dir, fname)
            torch.save(self.state_dict(), savepath)
            mlflow.log_artifact(savepath, artifact_path)


def train_step(model, batch, optimizer, mask):
    """
    Single step of training loop.

    Parameters
    ----------
    model: torch.nn.Module object
        model to train

    batch: torch_geometric.data.Data object
        data to train on

    optimizer: torch.optim.Optimizer object
        optimizer for model

    mask: torch.Tensor
        [N-node] element boolean tensor.
        If mask[n] == True, loss from prediction on node n is used to update gradients during training.
        Otherwise, the prediction is ignored during training.

    Returns
    -------
    None

    Other
    -------
    Updates model parameters in-place.
    """

    model.train()  # set training mode to True (enable dropout, etc)
    optimizer.zero_grad()  # reset gradients

    log_softmax = model(batch)  # call forward on model
    labels = batch.y.long()  # ground truth y labels (as class idx, NOT one-hot encoded)

    # torch.nn.functional.nll_loss(input, target)
    # input: NxC double tensor. input[i,j] contains the log-probability (ie log-softmax)
    #                             prediction for class j of sample i
    # target: N-element int tensor N[i] gives the ground truth class label for sample i
    # forward() needs all nodes for message passing
    # however, when computing loss, we only consider nodes included in mask to avoid including unwanted nodes
    # (ie validation/test nodes, non-candidate grains, etc)
    nll_loss = F.nll_loss(log_softmax[mask], labels[mask])
    nll_loss.backward()  # compute gradients
    optimizer.step()  # update model params
    
    return


def train_loop(model, data_train, data_valid, data_test, optimizer, max_iterations,
               checkpoint_iterations):
    """
    Training a single model with a fixed set of parameters.

    Logs various parameters to mlflow tracking server


    Parameters
    ----------
    model: torch.nn.Module object
        model to train

    data_train, data_valid, data_test: torch_geometric.data.Data
        training, validation, testing data to train/evaluate model on
        test data only used for final model

    optimizer: torch.optim.Optimizer object
        optimizer for model

    max_iterations: int

    checkpoint_iterations

    Returns
    --------
    best_model: OrderedDict
        state dict (model.state_dict()) from model with lowest validation loss

    """
    print("Training model")
    
    # store initial params and initial accuracy/loss for nicer training graphs
    ta, tl = mean_acc_and_loss(model, data_train, data_train.candidate_mask)  # train accuracy, loss
    va, vl = mean_acc_and_loss(model, data_valid, data_valid.candidate_mask)  # val accuracy, loss

    best_params = model.state_dict()
    # save initial weights
    model.save_state('{:04}.pt'.format(0))
    mlflow.log_metrics({'fit_train_acc': ta, 'fit_train_loss': tl,
                        'fit_val_acc': va, 'fit_val_loss': vl}, step=0)
    best_val_loss = 1e10
    best_iter = 0
    msg = "  iteration {:>4d}/{:<4d} Train acc: {:.4f}, Val acc: {:.4f}, Train loss: {:.4f}, Val loss: {:.4f}"
    # train model. Store train/val acc/loss at desired checkpoint iterations
    iters = range(checkpoint_iterations, max_iterations + 1, checkpoint_iterations)
    t = tqdm(iters, desc=msg.format(0, max_iterations, ta, va, tl, vl), leave=True)
    for train_iter in t:
        for _ in range(checkpoint_iterations):  # train for number of iterations in each checkpoint period
            train_step(model, data_train, optimizer, data_train.candidate_mask)

        # at the end of the checkpoint period, record loss and accuracy metrics
        ta, tl = mean_acc_and_loss(model, data_train, data_train.candidate_mask)  # train accuracy, loss
        va, vl = mean_acc_and_loss(model, data_valid, data_valid.candidate_mask)  # val accuracy, loss
        model.save_state("{:>04}.pt".format(train_iter))
        mlflow.log_metrics({'fit_train_acc': ta, 'fit_train_loss': tl,
                            'fit_val_acc': va, 'fit_val_loss': vl}, step=train_iter)

        # update progress bar
        t.set_description(msg.format(train_iter, max_iterations, ta, va, tl, vl))
        t.refresh()

        # if validation loss is lower than previous best, update best val loss and model params
        if vl < best_val_loss:
            best_iter = train_iter
            best_val_loss = vl
            best_train_loss = tl
            best_train_acc = ta
            best_val_acc = va
            best_params = model.state_dict()


    print('\nlogging model')
    # logging best model
    model.load_state_dict(best_params)
    ypt = model.predict(data_train)[data_train.candidate_mask]  # y pred train
    ypv = model.predict(data_valid)[data_valid.candidate_mask]  # y pred validataion
    gtt = data_train.y[data_train.candidate_mask]  # ground truth train
    gtv = data_valid.y[data_valid.candidate_mask]  # ground truth validation

    test_acc, test_loss = mean_acc_and_loss(model, data_test, data_test.candidate_mask)

    best_metrics = {'best_iter': best_iter,
                        'train_acc': best_train_acc,
                        'train_loss': best_train_loss,
                        'val_acc': best_val_acc,
                        'val_loss': best_val_loss,
                        'test_acc': test_acc,
                        'test_loss': test_loss}
    mlflow.log_metrics(best_metrics)  # log iteration with lowest val loss

    return best_params, best_metrics


@torch.no_grad()  # speed up predictions by disabling gradients
def mean_acc_and_loss(model, data, mask):
    """
    Computes mean accuracy and nll loss of samples after applying mask
    to select nodes to include in calculations.

    Parameters
    ----------
----------
    model: torch.nn.Module object
        model to train

    data: torch_geometric.data.Data object
        data to evaluate model on
    y: Tensor
        targets of ground truth values for each item in data


    mask: torch.Tensor
        [N-node] element boolean tensor.
        If mask[n] == True, loss from prediction on node n is included in accuracy and loss calculations.
        Otherwise, the prediction for node n is ignored.

    Returns
    -------
    acc, loss: float (may be 0-dim torch tensor)
        mean accuracy and nll-loss of predictions
    """
    model.train(False)
    # note: avoid using model.predict() to avoid calling forward twice (we also need log probabilities for loss)
    log_softmax = model(data)[mask]

    yp = log_softmax.argmax(dim=1)  # predicted log-probabilities
    yt = data.y[mask].long()  # ground truth class labels (int, not one-hot)

    nll_loss = float(F.nll_loss(log_softmax, yt))  # nll_loss
    acc = float((yp == yt).sum() / len(yt))  # mean accuracy

    return acc, nll_loss
