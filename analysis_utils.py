import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

get_L1_norm = lambda s, r: np.mean(np.abs((s -  r)))
get_L2_norm = lambda s, r: np.mean((s -  r)**2)**.5
get_L4_norm = lambda s, r: np.mean((s -  r)**4)**.25

get_stats = lambda s, r: np.array((get_L1_norm(s, r), get_L2_norm(s, r), get_L4_norm(s, r)))
get_agg_stats = lambda s, rs: np.mean([get_stats(s, r) for r in rs], axis=0)
format_stats = lambda l1, l2, l4: "{: 6.4f} {:6.4f} {:6.4f}".format(l1, l2, l4)

def plot_mean_w_error(data, axis=1, label=None):
    means = data.mean(axis=axis)
    stds = data.std(axis=axis)
    plt.plot(means, label=label)
    plt.fill_between(range(len(means)), means-stds, means+stds, alpha=.75)

def plot_window(x, ys, labels=None, colors=None, window=None, alpha=1):
    w = window if window is not None else (0, len(x))
    ls = labels if labels is not None else len(ys) * [None]
    cs = colors if colors is not None else plt.cm.viridis(np.linspace(0, 1, len(ys)))
    for i in range(len(ys)):
        x_slice = x[w[0]:w[1]]
        y_slice = np.squeeze(ys[i])[w[0]:w[1]]
        plt.plot(x_slice, y_slice, label=ls[i], color=cs[i], alpha=alpha)

def find_spectrum(label, batches, sess, input_stream, label_stream):
    for iteration in tqdm(batches):
        spectra = sess.run(input_stream)
        labels = sess.run(label_stream)
        for i, l in enumerate(range(labels)):
            if np.all(l == label):
                return spectra[i]
    return None

def find_spectra(label, batches, sess, input_stream, label_stream):
    # nans are treated as wildcards
    label_np = np.array(label).astype('float32')
    matching_spectra = []
    matching_labels = []
    for iteration in tqdm(range(batches)):
        spectra = sess.run(input_stream)
        labels = sess.run(label_stream)
        for i, l in enumerate(labels):
            if np.logical_or(l == label_np, np.isnan(label_np)).all():
                matching_spectra.append(spectra[i])
                matching_labels.append(l)
    return np.array(matching_spectra), np.array(matching_labels)

def sort_spectra(spectra, labels, index):
    sorted_indices = labels[:, index].argsort()
    sorted_labels = labels[sorted_indices]
    sorted_spectra = spectra[sorted_indices]
    return sorted_spectra, sorted_labels
