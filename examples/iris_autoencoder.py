""" A demonstration of how to use the Deep Autoencoder. Using the small iris dataset.
"""



from sklearn import datasets
from skflow import StackedAutoEncoder

iris = datasets.load_iris()
x = iris.data

model = StackedAutoEncoder(dims=[5,6], activations=['relu', 'relu'], noise='gaussian', epoch=10000,
                            loss='rmse', lr=0.007, batch_size=50, print_step=2000)
pp = model.fit_transform(x)
