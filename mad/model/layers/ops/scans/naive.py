import torch

def naive_pscan(A, X, Y_init):
    y = Y_init
    scan = []

    for k in range(A.size(1)):
        y = A[:, k, :] * y + X[:, k, :]
        scan.append(y)
    Y_ = torch.stack(scan,dim=1)
    return Y_

fn = naive_pscan