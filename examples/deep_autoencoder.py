""" A demonstration of how to use the Deep Autoencoder. Using the small iris dataset.
"""



from sklearn import datasets
import skflow
iris = datasets.load_iris()
x = iris.data

result = skflow.StackedAutoEncoder(x, dims=[5, 4, 3], noise='gaussian', epoch=1000).encode()
print(result)
