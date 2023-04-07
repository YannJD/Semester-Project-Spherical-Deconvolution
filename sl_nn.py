import torch
import torch.nn as nn
import numpy as np
import time
from cmath import inf


class RegularizedMSE(nn.Module):
    def __init__(self, kernel, reg_B, reg_factor, device):
        super(RegularizedMSE, self).__init__()
        self.kernel = torch.tensor(kernel, dtype=torch.float32)
        self.reg_B = torch.tensor(reg_B, dtype=torch.float32)
        self.reg_factor = reg_factor
        self.device = device

    def forward(self, output, target):
        criterion = nn.MSELoss()
        H_f = torch.matmul(self.kernel.to(self.device), output.t().to(self.device)).t()
        loss = criterion(H_f, target)
        B_f = torch.matmul(self.reg_B.to(self.device), output.t().to(self.device)).t()
        B_f = torch.minimum(B_f, torch.zeros(B_f.size()).to(self.device))
        neg_constraint = torch.sum(B_f)
        loss += self.reg_factor * torch.square(neg_constraint)
        return loss


class ConstrainedMSE(nn.Module):
    def __init__(self, kernel, B, M, device):
        super(ConstrainedMSE, self).__init__()
        self.kernel = torch.tensor(kernel, dtype=torch.float32)
        self.B = torch.tensor(B, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)
        self.device = device

    def forward(self, output, target):
        criterion = nn.MSELoss()

        B_f = torch.matmul(self.B.to(self.device), output.t().to(self.device)).t()
        B_f[B_f < 0] = 0
        new_f = torch.matmul(self.M.to(self.device), B_f.t())
        H_f = torch.matmul(self.kernel.to(self.device), new_f).t()

        return criterion(H_f, target)


def create_nn_arch(arch: np.ndarray):
    layers = []

    for in_size, out_size in zip(arch[:-2], arch[1:-1]):
        layers.append(nn.Linear(in_size, out_size))
        #layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(out_size))
        layers.append(nn.Sigmoid())

    layers.append(nn.Linear(arch[-2], arch[-1]))
    #layers.append(nn.ReLU())
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


class sl_nn(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.network = create_nn_arch(arch)

    def forward(self, x):
        return self.network(x)

    def evaluate_odf_sh(self, signal, device=torch.device("cpu")):
        signal = torch.tensor(signal, dtype=torch.float32).to(device)
        self.eval()
        with torch.no_grad():
            self.to(device)
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
            f"Epoch {epoch + 1:2}:   learning rate = {optimizer.param_groups[0]['lr']:>0.6f}   average loss = {loss_epoch[-1]:0.6f}")
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
