import sys
sys.path.insert(0, "..")
from tensorflow import keras
from kdg.utils import generate_polynomial as generate_data
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.io import savemat, loadmat
import argparse

from kdn_versions.kdn_all_w_apprx_lin_npar import kdn as kdn1
from kdn_versions.kdn_all_w_strict_lin_npar import kdn_all_w as kdn2
from kdn_versions.kdn_all_w_meta_poly_npar import kdn_all_w_meta_poly as kdn3

parser = argparse.ArgumentParser()
parser.add_argument('-kdnversion')
parser.add_argument('-c')
parser.add_argument('-k')
parser.add_argument('-reps')
args = vars(parser.parse_args())
selectKDN = int(args['kdnversion'])
c = float(args['c'])
k = float(args['k'])
reps = int(args['reps'])
print("Running the Polynomial experiment...")

# select the KDN to run the experiments
# 1: KDN approx lin, 2: KDN strict lin, 3: KDN meta poly

kdn_versions = [
    "KDN_approx_lin",
    "KDN_strict_lin",
    "KDN_meta_poly"
]

# kdn approx lin
kdn1_kwargs = {
    "k": k,
    "polytope_compute_method": "all",
    "weighting_method": "lin",
    "T": 2,
    "c": c,
    "verbose": False
}

# kdn strict lin
kdn2_kwargs = {
    "k": k,
    "polytope_compute_method": "all",
    "weighting_method": "lin",
    "T": 2,
    "c": c,
    "verbose": False
}

# kdn meta poly
kdn3_kwargs = {
    "k": k,
    "polytope_compute_method": "all",
    "weighting_method": "lin",
    "T": 2,
    "c": c,
    "verbose": False
}

# Define NN parameters
X_val, y_val = generate_data(500, a=(1, 3))
compile_kwargs = {
    "loss": "binary_crossentropy",
    "optimizer": keras.optimizers.Adam(3e-4),
}
callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=False)
fit_kwargs = {
    "epochs": 300,
    "batch_size": 64,
    "verbose": False,
    "validation_data": (X_val, keras.utils.to_categorical(y_val)),
    "callbacks": [callback],
}

# network architecture [10, 10, 10, 10, 2]
def getNN():
    network_base = keras.Sequential()
    initializer = keras.initializers.GlorotNormal(seed=0)
    network_base.add(keras.layers.Dense(10, activation="relu", kernel_initializer=initializer, input_shape=(2,)))
    network_base.add(keras.layers.Dense(10, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(10, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(10, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(units=2, activation="softmax", kernel_initializer=initializer))
    network_base.compile(**compile_kwargs)
    return network_base

# create out-of-distribution samples
def generate_ood_samples(n, inbound=[1, -1], outbound=[5, -5]):
    Xood = []
    i = 0
    while True:
        x1 = (outbound[0] - outbound[1])*np.random.random_sample() - outbound[0]
        x2 = (outbound[0] - outbound[1])*np.random.random_sample() - outbound[0]
        if (-inbound[0] < x1 < inbound[1]) and (-inbound[0] < x2 < inbound[1]):
            continue
        else:
            Xood.append([x1, x2])
            i += 1
        if i >= n:
            break
    Xood = np.array(Xood)
    return Xood

def hellinger_distance(p, q):
   """Hellinger distance between two discrete distributions.
      Same as original version but without list comprehension
   """
   return np.mean(np.linalg.norm(np.sqrt(p)-np.sqrt(q), ord=2, axis=1))/np.sqrt(2)

def compute_stats(param):
    return np.median(param), np.quantile(param, [0.25])[0], np.quantile(param, [0.75])[0]

# define the grid
p = np.arange(-1, 1, step=0.01)
q = np.arange(-1, 1, step=0.01)
xx, yy = np.meshgrid(p, q)
tmp = np.ones(xx.shape)
grid_samples = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)

# get true posterior
tp_df = pd.read_csv("tp.csv")
tmp = np.array([tp_df["posterior"][x] for x in range(40000)])
tmp = np.flip(tmp.reshape(200, 200), axis=1)
proba_true = tmp

# run the experiment
sample_size = [10, 50, 100, 500, 1000, 5000, 10000]
n_test = 1000

accuracy_nn = []
accuracy_kdn = []

mmcIn_nn = []
mmcIn_kdn = []

mmcOut_nn = []
mmcOut_kdn = []

hd_nn = []
hd_kdn = []

reps_list = []
sample_list = []
ddf = {}

true_pos = np.vstack((proba_true.ravel(), 1 - proba_true.ravel())).T # true posterior over a [-2, 2] grid

# run experiment
for sample in sample_size:
    print("Doing sample %d" % sample)
    for ii in range(reps):
        X, y = generate_data(sample, a=(1, 3))
        X_test, y_test = generate_data(n_test, a=(1, 3))
        X_ood = generate_ood_samples(n_test)

        # train Vanilla NN
        nn = getNN()
        nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)

        accuracy_nn.append(
            np.mean(np.argmax(nn.predict(X_test), axis=1) == y_test)
        )
        mmcIn_nn.append(
            np.mean(np.max(nn.predict(X_test), axis=1))
        )
        mmcOut_nn.append(
            np.mean(np.max(nn.predict(X_ood), axis=1))
        )
        proba_nn = nn.predict(grid_samples)
        proba_nn = np.flip(proba_nn[:, 0].reshape(200, 200), axis=1)
        nn_pos = np.vstack((proba_nn.ravel(), 1 - proba_nn.ravel())).T
        hd_nn.append(
            hellinger_distance(nn_pos, true_pos)
        )
        
        # train KDN1
        if selectKDN == 1:
            model_kdn = kdn1(nn, **kdn1_kwargs)
        if selectKDN == 2:
            model_kdn = kdn2(nn, **kdn2_kwargs)
        if selectKDN == 3:
            model_kdn = kdn3(nn, **kdn3_kwargs)
        model_kdn.fit(X, y)

        accuracy_kdn.append(
            np.mean(model_kdn.predict(X_test) == y_test)
        )
        mmcIn_kdn.append(
            np.mean(np.max(model_kdn.predict_proba(X_test), axis=1))
        )
        mmcOut_kdn.append(
            np.mean(np.max(model_kdn.predict_proba(X_ood), axis=1))
        )
        proba_kdn = model_kdn.predict_proba(grid_samples)
        proba_kdn = np.flip(proba_kdn[:, 0].reshape(200, 200), axis=1)
        kdn_pos = np.vstack((proba_kdn.ravel(), 1 - proba_kdn.ravel())).T
        hd_kdn.append(
            hellinger_distance(kdn_pos, true_pos)
        )
        
        reps_list.append(ii)
        sample_list.append(sample)

ddf["kdn_acc"] = accuracy_kdn
ddf["nn_acc"] = accuracy_nn

ddf["kdn_mmcIn"] = mmcIn_kdn
ddf["nn_mmcIn"] = mmcIn_nn

ddf["kdn_mmcOut"] = mmcOut_kdn
ddf["nn_mmcOut"] = mmcOut_nn

ddf["kdn_hd"] = hd_kdn
ddf["nn_hd"] = hd_nn

ddf["reps"] = reps_list
ddf["sample"] = sample_list

# plot the curves (without errorbars)
sample_size = [10, 50, 100, 500, 1000, 5000, 10000]
metrics = ["acc", "mmcIn", "mmcOut", "hd"]
names = ["nn", "kdn"]
colors = ["k", "r"]
labels = ["NN", "KDN"]
fig3, axes = plt.subplots(1, 4, figsize=(20, 5))
for k, metric in enumerate(metrics):
    ax = axes[k]
    for i, name in enumerate(names):
        param_med = []
        param_25_quantile = []
        param_75_quantile = []

        for sample in sample_size:
            if metric == "acc":
                param = 1 - np.array(ddf[name + "_" + metric])[np.array(ddf["sample"]) == sample]
            else:
                param = np.array(ddf[name + "_" + metric])[np.array(ddf["sample"]) == sample]
            s1, s2, s3 = compute_stats(param)
            param_med.append(s1)
            param_25_quantile.append(s2)
            param_75_quantile.append(s3)

        ax.plot(sample_size, param_med, c=colors[i], label=labels[i])
        ax.fill_between(
            sample_size, param_25_quantile, param_75_quantile, facecolor= colors[i], alpha=0.3
        )
        
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)

    if metric == "acc":
        ax.set_xscale("log")
        ax.set_ylim([0, 0.6])
        ax.set_ylabel("Error", fontsize=15)
    if metric == "mmcIn":
        ax.set_ylim([0, 1])
        ax.set_ylabel("Mean Max Confidence (In Distribution)", fontsize=15)    
    if metric == "mmcOut":
        ax.set_ylim([0, 1])
        ax.set_ylabel("Mean Max Confidence (Out Distribution)", fontsize=15) 
    if metric == "hd":
        ax.set_ylim([0, 1])
        ax.set_ylabel("Hellinger Distance", fontsize=15) 
    ax.set_xlabel("Sample Size", fontsize=15)
    ax.legend(frameon=False, fontsize=15)
    

name = kdn_versions[selectKDN-1]
filename = "results/" + name + "_experiment_data.mat"
savemat(filename, ddf)

filename = "plots/" + name + "_results_plots.pdf"
fig3.savefig(filename, bbox_inches='tight')