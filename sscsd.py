import torch
import torch.nn as nn
import numpy as np
import time
from cmath import inf
from qpth.qp import QPFunction
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class RegularizedLoss(nn.Module):
    """
    Regularized version of the MSE loss for CSD
    """

    def __init__(self, criterion, kernel, reg_B, reg_factor, device):
        super(RegularizedLoss, self).__init__()
        self.criterion = criterion
        self.kernel = torch.tensor(kernel, dtype=torch.float32)
        self.reg_B = torch.tensor(reg_B, dtype=torch.float32)
        self.reg_factor = reg_factor
        self.device = device

    def forward(self, output, target):
        # Computes H dot f and standard loss between signal and the result
        H_f = torch.matmul(self.kernel.to(self.device), output.t().to(self.device)).t()
        loss = self.criterion(H_f, target)

        # Computes the following expression to penalize negative fFODs: loss + lambda * sum(min(B @ f))^2
        B_f = torch.matmul(self.reg_B.to(self.device), output.t().to(self.device)).t()
        B_f = torch.minimum(B_f, torch.zeros(B_f.size()).to(self.device))
        neg_constraint = torch.sum(B_f)
        loss += self.reg_factor * torch.square(neg_constraint)
        return loss


def create_nn_arch(arch: np.ndarray):
    """
    Creates a network architecture.

    :param arch: array
    :return: the architecture
    """

    layers = []

    # Adds for each network layer the following activation functions
    for in_size, out_size in zip(arch[:-2], arch[1:-1]):
        layers.append(nn.Linear(in_size, out_size))
        # layers.append(nn.BatchNorm1d(out_size))
        # layers.append(nn.Dropout(p=0.5))
        # layers.append(nn.ReLU())
        layers.append(nn.GELU())
        # layers.append(nn.Sigmoid())

    layers.append(nn.Linear(arch[-2], arch[-1]))
    # layers.append(nn.ReLU())
    # layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


class SSCSD(nn.Module):
    """
    Self-supervised MLP for CSD fODF estimation
    """

    def __init__(self, arch, H, B, M):
        super().__init__()
        self.network = create_nn_arch(arch)
        self.H = torch.tensor(H, dtype=torch.float32)
        self.B = torch.tensor(B, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)
        self.G = Parameter(torch.tensor(-B, dtype=torch.float32))
        self.Q = Variable(torch.eye(arch[-1]).cuda())
        self.h = Parameter(torch.zeros(B.shape[0]).cuda())

    def forward(self, x):
        # p = self.network(x)
        # e = Variable(torch.Tensor())
        # [Q, p, G, h, A, b] = [Variable(x.data.double().cuda()) for x in [self.Q, p, self.G, self.h, e, e]]
        # x = QPFunction(verbose=False)(Q, p, G, h, A, b)
        # return x.data.float()

        # B_f = torch.matmul(self.B.cuda(), self.network(x).t())
        # B_f[B_f < 0] = 0
        # return torch.matmul(self.M.cuda(), B_f).t()
        return self.network(x)

    def evaluate_odf_sh(self, signal, device=torch.device("cpu")):
        """
        Evaluates the fODF SH coefficient from the signal.

        :param signal: 1D array
        :param device: str
        :return: the fODF SH coefficients estimation
        """

        signal = torch.tensor(signal, dtype=torch.float32).to(device)
        min_signal = torch.min(signal, 1)[0][..., None]
        max_signal = torch.max(signal, 1)[0][..., None]
        self.eval()
        with torch.no_grad():
            self.to(device)
            # return self.forward(signal) * (max_signal - min_signal)
            return self.forward(signal)


def train_model(
        model,
        device,
        train_loader,
        loss_fun,
        optimizer,
        lr_sched,
        epochs,
        load_best_model=True,
        return_loss_time=False
):
    """
    Train self-supervised MLP.

    :param model: SSCSD MLP
    :param device: str
    :param train_loader: DataLoader
    :param loss_fun: the loss function
    :param optimizer: the optimizer
    :param lr_sched: learning rate scheduler
    :param epochs: the number of epochs, int
    :param load_best_model: bool
    :param return_loss_time: bool
    :return: all the losses, the best epoch and the training time
    """

    start = time.time()
    loss_epoch = []
    best_loss = torch.tensor(inf)
    best_epoch = epochs

    for epoch in range(epochs):
        model.train()

        for batch, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            f_sh = model(xb).to(device)

            loss_batch = []
            loss = loss_fun(f_sh, yb)
            loss_batch.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_epoch.append(sum(loss_batch) / len(loss_batch))
        print(
            f"Epoch {epoch + 1:2}:   learning rate = {optimizer.param_groups[0]['lr']:>0.10f}   average loss = {loss_epoch[-1]:0.6f}")
        lr_sched.step(loss)

        if loss_epoch[-1] < best_loss:
            best_loss = loss_epoch[-1]
            best_epoch = epoch + 1
            best_state_dict = model.state_dict()

    if load_best_model:
        model.load_state_dict(best_state_dict)
        print(f"best model found at epoch = {best_epoch} is loaded")

    end = time.time()
    elapsed_time = end - start
    print(f'Total training time: {elapsed_time:0.3f} seconds')

    if return_loss_time:
        return loss_epoch, best_epoch, elapsed_time


def minMaxNormalization(x):
    x_min = np.amin(x, axis=1)[..., np.newaxis]
    x_max = np.amax(x, axis=1)[..., np.newaxis]
    return (x - x_min) / (x_max - x_min), x_min, x_max
