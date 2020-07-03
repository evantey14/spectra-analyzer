import numpy as np
import matplotlib.pyplot as plt

get_L1_norm = lambda s, r: np.abs(np.mean((s -  r)))
get_L2_norm = lambda s, r: np.mean((s -  r)**2)**.5
get_L4_norm = lambda s, r: np.mean((s -  r)**4)**.25

get_stats = lambda s, r: (get_L1_norm(s, r), get_L2_norm(s, r), get_L4_norm(s, r))
get_agg_stats = lambda s, rs: np.mean([get_stats(s, r) for r in rs], axis=0)
format_stats = lambda l1, l2, l4: "{: 6.4f} {:6.4f} {:6.4f}".format(l1, l2, l4)

def plot_mean_w_error(data, axis=1, label=None):
    means = data.mean(axis=axis)
    stds = data.std(axis=axis)
    plt.plot(means, label=label)
    plt.fill_between(range(len(means)), means-stds, means+stds, alpha=.75)

def plot_window(x, ys, labels=None, window=None, alpha=1):
    w = window if window is not None else (0, len(l))
    ls = labels if labels is not None else len(ys) * [None]
    for i in range(len(ys)):
        x_slice = x[w[0]:w[1]]
        y_slice = np.squeeze(ys[i])[w[0]:w[1]]
        plt.plot(x_slice, y_slice, label=ls[i], alpha=alpha)
