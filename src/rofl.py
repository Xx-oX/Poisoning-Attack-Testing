import numpy as np
import torch

def norm_bound(param, norm=None):
    """
    Check if the norm of w is bounded by B.
    Version without ZKP
    param: local update parameter (saved by state_dict)
    norm: "inf" for infinity norm, else: L2 norm
    """
    w = []
    for name in param.keys():
        # print(name, np.average(param[name].data.cpu().numpy().flatten()))
        if ("weight" in name):
            w.append(param[name].data.cpu().numpy().flatten())
    h = np.hstack(w)
    w_q = probabilistic_quantization(h, 16)
    # print("L2 norm: ", np.linalg.norm(w_q, ord=2))
    # print("L-inf norm: ", np.linalg.norm(w_q, ord=np.inf))
    if norm == "inf":
        return np.linalg.norm(w_q, ord=np.inf)
    else:
        return np.linalg.norm(w_q, ord=2)

def probabilistic_quantization(h, b):
    """
    Perform b-bit probabilistic quantization.
    h: consider the update H, let h = (h1, h2, ...) = vec(H)
    b: b-bit quantization
    """
    result = np.zeros(h.size)
    h_min, h_max = np.min(h), np.max(h)
    interval = (h_max - h_min) / (2 ** b)

    for j, h_j in enumerate(h):
        if h_j == h_min:
            result[j] = h_min
            continue
        if h_j == h_max:
            result[j] = h_max
            continue

        # h_l: lower bound of the interval
        # h_u: upper bound of the interval
        h_l = h_min + interval * np.floor((h_j - h_min) / interval)
        h_u = h_l + interval

        # h_j = h_l with probability (h_j - h_l) / h_u - h_l
        # h_j = h_u with probability (h_u - h_j) / h_u - h_l
        if np.random.rand() < (h_j - h_l) / (h_u - h_l):
            result[j] = h_l
        else:
            result[j] = h_u
    return result
