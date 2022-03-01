#%%
from kdg import kdn
from kdg.utils import get_ece
from tensorflow import keras
import openml
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece
import os
from os import listdir, getcwd 
# %%

# network architecture [10, 10, 10, 10, 2]
def getNN(compile_kwargs, num_classes):
    network_base = keras.Sequential()
    initializer = keras.initializers.GlorotNormal(seed=0)
    network_base.add(keras.layers.Dense(50, activation="relu", kernel_initializer=initializer, input_shape=(2,)))
    network_base.add(keras.layers.Dense(50, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(50, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(50, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(units=num_classes, activation="softmax", kernel_initializer=initializer))
    network_base.compile(**compile_kwargs)
    return network_base

def experiment(task_id, folder, n_estimators=500, reps=30):
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()

    if np.isnan(np.sum(y)):
        return

    if np.isnan(np.sum(X)):
        return

    total_sample = X.shape[0]
    unique_classes, counts = np.unique(y, return_counts=True)

    test_sample = min(counts)//3

    indx = []
    for label in unique_classes:
        indx.append(
            np.where(
                y==label
            )[0]
        )

    max_sample = min(counts) - test_sample
    train_samples = np.logspace(
        np.log10(2),
        np.log10(max_sample),
        num=3,
        endpoint=True,
        dtype=int
        )
    
    err = []
    err_rf = []
    ece = []
    ece_rf = []
    kappa = []
    kappa_rf = []
    mc_rep = []
    samples = []

    for train_sample in train_samples:
        
        for rep in range(reps):
            indx_to_take_train = []
            indx_to_take_test = []

            for ii, _ in enumerate(unique_classes):
                np.random.shuffle(indx[ii])
                indx_to_take_train.extend(
                    list(
                            indx[ii][:train_sample]
                    )
                )
                indx_to_take_test.extend(
                    list(
                            indx[ii][-test_sample:counts[ii]]
                    )
                )
            # Define NN parameters
            compile_kwargs = {
                "loss": "binary_crossentropy",
                "optimizer": keras.optimizers.Adam(3e-4),
            }
            fit_kwargs = {
                "epochs": 300,
                "batch_size": 64,
                "verbose": False,
            }
            kdn_kwargs = {
                "k": 1e-5,
                "polytope_compute_method": "all",
                "weighting_method": "lin",
                "T": 1e-3,
                "h": 1/2,
                "verbose": False
            }
            # train Vanilla NN
            nn = getNN(compile_kwargs, len(unique_classes))
            nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)

            model_kdn = kdn(nn, **kdn_kwargs)
            model_kdn.fit(X[indx_to_take_train], y[indx_to_take_train])
            proba_kdf = model_kdn.predict_proba(X[indx_to_take_test])
            proba_rf = model_kdn.predict_proba_nn(X[indx_to_take_test])
            predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
            predicted_label_rf = np.argmax(proba_rf, axis = 1)

            err.append(
                1 - np.mean(
                        predicted_label_kdf==y[indx_to_take_test]
                    )
            )
            err_rf.append(
                1 - np.mean(
                    predicted_label_rf==y[indx_to_take_test]
                )
            )
            kappa.append(
                cohen_kappa_score(predicted_label_kdf, y[indx_to_take_test])
            )
            kappa_rf.append(
                cohen_kappa_score(predicted_label_rf, y[indx_to_take_test])
            )
            ece.append(
                get_ece(proba_kdf, predicted_label_kdf, y[indx_to_take_test])
            )
            ece_rf.append(
                get_ece(proba_rf, predicted_label_rf, y[indx_to_take_test])
            )
            samples.append(
                train_sample*len(unique_classes)
            )
            mc_rep.append(rep)

    df = pd.DataFrame() 
    df['err_kdf'] = err
    df['err_rf'] = err_rf
    df['kappa_kdf'] = kappa
    df['kappa_rf'] = kappa_rf
    df['ece_kdf'] = ece
    df['ece_rf'] = ece_rf
    df['rep'] = mc_rep
    df['samples'] = samples

    df.to_csv(folder+'/'+'openML_cc18_'+str(task_id)+'.csv')

#%%
folder = 'openml_res'
os.mkdir(folder)
benchmark_suite = openml.study.get_suite('OpenML-CC18')
current_dir = getcwd()
files = listdir(current_dir+'/'+folder)
Parallel(n_jobs=10,verbose=1)(
        delayed(experiment)(
                task_id,
                folder
                ) for task_id in benchmark_suite.tasks
            )

'''for task_id in benchmark_suite.tasks:
    filename = 'openML_cc18_' + str(task_id) + '.csv'
    if filename not in files:
        print(filename)
        try:
            experiment(task_id,folder)
        except:
            print("couldn't run!")
        else:
            print("Ran successfully!")'''
# %%