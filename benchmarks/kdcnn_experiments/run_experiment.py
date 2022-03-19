# %%
import numpy as np
from architectures import getCNN
from data import getDataset
from kdg.kdcnn import *
from sklearn.utils import shuffle

x_train, y_train, x_test, y_test = getDataset(name='mnist').get_data()
cnn = getCNN(input_shape=(28, 28, 1)).LeNet()
cnn.summary()

fit_kwargs = {
    "epochs": 3,
    "batch_size": 256,
    "verbose": True
    }
cnn.fit(x_train, y_train, **fit_kwargs)

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)
X, y = x_train[:5000], y_train[:5000]
Xt, yt = x_test[:1000], y_test[:1000]

# %%

model_kdcnn = kdcnn(
    network=cnn,
    num_fc_layers=3,
    k=1e-5,
    h=1,
    T=1e-3,
    verbose=False
)

# %%
model_kdcnn.fit(X, y)

## Evaluate Vanilla CNN
proba_nn = cnn.predict(Xt)
predicted_label = np.argmax(proba_nn, axis=1)
accuracy = np.mean(predicted_label==yt.squeeze())
print("Accuracy CNN: {}".format(accuracy))

## Evaluate KD-CNN
y_pred = model_kdcnn.predict(Xt)
accuracy = np.mean(y_pred==yt.squeeze())
print("Accuracy KDCNN: {}".format(accuracy))




# %%
