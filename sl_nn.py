import torch
import torch.nn as nn
import numpy as np
import time
from cmath import inf


def create_nn_arch(arch: np.ndarray):
    layers = []

    for in_size, out_size in zip(arch[:-2], arch[1:-1]):
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.BatchNorm1d(out_size))
        layers.append(nn.Sigmoid())

    layers.append(nn.Linear(arch[-2], arch[-1]))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


class sl_nn(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.network = create_nn_arch(arch)

    def forward(self, x):
        return self.network(x)

    def evaluate_fod_sh(self, signal, device=torch.device("cpu")):
        signal = torch.tensor(signal, dtype=torch.float32).to(device)
        self.eval()
        with torch.no_grad():
            self.to(device)
            return self.forward(signal)


def train_model(
        model,
        kernel,
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

            f_sh = model(xb)

            loss_batch = []
            loss = loss_fun(yb, torch.matmul(kernel, f_sh.t()).t())
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
