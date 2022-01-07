#
# Created on Thu Dec 09 2021 6:04:08 AM
# Author: Ashwin De Silva (ldesilv2@jhu.edu)
# Objective: Polynomial Experiment
#
#%%
# import standard libraries
import numpy as np
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# import internal libraries
from kdg.kdn import *
from kdg.utils import generate_polynomial
# %%

# generate training data
X, y = generate_polynomial(10000, a=(1, 3))
X_val, y_val = generate_polynomial(500, a=(1, 3))

# NN params
compile_kwargs = {
    "loss": "binary_crossentropy",
    "optimizer": keras.optimizers.Adam(3e-4),
}
callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=True)
fit_kwargs = {
    "epochs": 300,
    "batch_size": 32,
    "verbose": True,
    "validation_data": (X_val, keras.utils.to_categorical(y_val)),
    "callbacks": [callback],
}

# network architecture
def getNN():
    network_base = keras.Sequential()
    network_base.add(keras.layers.Dense(6, activation="relu", input_shape=(2,)))
    network_base.add(keras.layers.Dense(6, activation="relu"))
    network_base.add(keras.layers.Dense(6, activation="relu"))
    # network_base.add(keras.layers.Dense(5, activation="relu"))
    network_base.add(keras.layers.Dense(units=2, activation="softmax"))
    network_base.compile(**compile_kwargs)
    return network_base


# train Vanilla NN
vanilla_nn = getNN()
history = vanilla_nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)

# plot the training loss and validation loss
fig, ax = plt.subplots()
ax.plot(history.history["loss"])
ax.plot(history.history["val_loss"])
ax.set_xlabel("epochs")
ax.set_ylabel("loss")
ax.legend(["train", "val"])

# print the accuracy of Vanilla NN and KDN
X_test, y_test = generate_polynomial(1000, a=(1, 3))
accuracy_nn = np.mean(np.argmax(vanilla_nn.predict(X_test), axis=1) == y_test)
print("Vanilla NN accuracy : ", accuracy_nn)

# %%
# train KDN
model_kdn = kdn(
    network=vanilla_nn,
    k=1e-5,
    polytope_compute_method="all",
    weighting_method="lin",
    T=2,
    c=2,
    verbose=False,
)
model_kdn.fit(X, y)

# print the accuracy of Vanilla NN and KDN
accuracy_kdn = np.mean(model_kdn.predict(X_test) == y_test)
print("KDN accuracy : ", accuracy_kdn)

# plot

# %%
# define the grid
p = np.arange(-1.5, 1.5, step=0.005)
q = np.arange(-1.5, 1.5, step=0.005)
xx, yy = np.meshgrid(p, q)
tmp = np.ones(xx.shape)
grid_samples = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)

# plot
# proba_kdn = model_kdn.predict_proba(grid_samples)
# proba_nn = model_kdn.predict_proba_nn(grid_samples)

# %%
filename = "results/polynomial.csv"
data = np.load( "results/polynomial.npz")
X = data['X']
y = data['y']
proba_nn = data['proba_nn']
proba_kdn = data['proba_kdn']
X, y = generate_polynomial(100000, a=(1, 3))

fig, ax = plt.subplots(1, 4, figsize=(40, 10))

import matplotlib
cmap = matplotlib.cm.get_cmap('bwr')
# colors = sns.color_palette("bwr", n_colors=2)
clr = [cmap(255 - 255*i) for i in y]
ax[0].scatter(X[:, 0], X[:, 1], c=clr, s=1, alpha=0.5)
ax[0].set_xlim(-1.5, 1.5)
ax[0].set_ylim(-1.5, 1.5)
ax[0].set_title("Data", fontsize=24)
ax[0].set_aspect("equal")
ax[0].tick_params(axis='both', labelsize=20)

ax1 = ax[1].imshow(
    np.flip(proba_nn[:, 0].reshape(600, 600), axis=1),
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest"
    # aspect="auto",
)
ax[1].set_title("NN", fontsize=24)
# ax[1].set_aspect("equal")
cbar = fig.colorbar(ax1, ax=ax[1], fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20) 
ax[1].tick_params(axis='both', labelsize=15)

ax2 = ax[2].imshow(
    np.flip(proba_kdn[:, 0].reshape(600, 600), axis=1),
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest"
    # aspect="auto",
)
ax[2].set_title("KDN", fontsize=24)
# ax[2].set_aspect("equal")
cbar = fig.colorbar(ax2, ax=ax[2], fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20) 
ax[2].tick_params(axis='both', labelsize=15)

df = pd.read_csv(filename)

sample_size = [5, 10, 50, 100, 500, 1000, 5000, 10000]

err_nn_med = []
err_nn_25_quantile = []
err_nn_75_quantile = []

err_kdn_med = []
err_kdn_25_quantile = []
err_kdn_75_quantile = []

for sample in sample_size:
    err_nn = 1 - df["accuracy nn"][df["sample"] == sample]
    err_kdn = 1 - df["accuracy kdn"][df["sample"] == sample]

    err_nn_med.append(np.median(err_nn))
    err_nn_25_quantile.append(np.quantile(err_nn, [0.25])[0])
    err_nn_75_quantile.append(np.quantile(err_nn, [0.75])[0])

    err_kdn_med.append(np.median(err_kdn))
    err_kdn_25_quantile.append(np.quantile(err_kdn, [0.25])[0])
    err_kdn_75_quantile.append(np.quantile(err_kdn, [0.75])[0])

ax[3].plot(sample_size, err_nn_med, c="k", label="NN")
ax[3].fill_between(
    sample_size, err_nn_25_quantile, err_nn_75_quantile, facecolor="k", alpha=0.3
)

ax[3].plot(sample_size, err_kdn_med, c="r", label="KDN")
ax[3].fill_between(
    sample_size, err_kdn_25_quantile, err_kdn_75_quantile, facecolor="r", alpha=0.3
)

right_side = ax[3].spines["right"]
right_side.set_visible(False)
top_side = ax[3].spines["top"]
top_side.set_visible(False)

ax[3].set_xscale("log")
ax[3].set_xlabel("Sample Size", fontsize=24)
ax[3].set_ylabel("Error", fontsize=24)
ax[3].legend(frameon=False, fontsize=20)
ax[3].tick_params(axis='both', labelsize=20)
asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
ax[0].set_aspect(asp)
# ax[3].set_aspect("equal")

fig.savefig("plots/polynomial.pdf")
plt.show()

# %%

# save the figure data
np.savez(
    "results/polynomial.npz",
    X=X, 
    y=y,
    proba_nn=proba_nn, 
    proba_kdn=proba_kdn
)
# %%
