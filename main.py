import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# function to visualize the digit images
from sklearn.neural_network import MLPClassifier


def plot_images(images, labels):
    n_cols = min(5, len(images))
    n_rows = len(images) // n_cols
    fig = plt.figure(figsize=(8, 8))

    for i in range(n_rows * n_cols):
        sp = fig.add_subplot(n_rows, n_cols, i+1)
        plt.axis("off")
        plt.imshow(images[i], cmap="gray")
        sp.set_title(labels[i])
    plt.show()

# get mnist dataset from openml
mnist = fetch_openml("mnist_784")
# set X to the data and y to the target values(answers)
X = mnist['data']
y = mnist['target']

# preprocessing
# X: values are between 0 and 255 we want to normalize them between 0 and 1 because standard practice
X = X / 255.0
# Y: floats are not needed, this removes the floatingpoint and decimal zeroes
y = y.astype("int32")

print("Dtypes:")
print(X.dtype, y.dtype)
print("Shape:")
print(X.shape, y.shape)

# x.dtype is an array that's 70'000 units long and every element is an array of 784
# the images are put into a 2 dimensional array and then flattened out into that vector
# since the images are 28x28px the vector is 784 units long

# shuffle the list, get the first 20 values(random)
'''p = np.random.permutation(len(X))
p = p[:20]
# visualize the values with the function above
plot_images(X[p].reshape(-1, 28, 28), y[p])'''

# split data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y)
print("Split Shapes:")
print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)

# train model
mlp = MLPClassifier(hidden_layer_sizes=(1000,),
                    max_iter=100,
                    alpha=1e-5,
                    solver='adam',
                    verbose=10,
                    tol=1e-4,
                    random_state=1,
                    learning_rate_init=.1)

mlp.fit(train_X, train_y)

# evaluations
# score
print()
print("Training set score: {0}".format(mlp.score(train_X, train_y)))
print("Test set score: {0}".format(mlp.score(test_X, test_y)))

mlp.score(train_X, train_y)

predictions = mlp.predict(test_X)
print(classification_report(test_y, predictions))

# plot predictions
p = np.random.permutation(len(test_X))
p = p[:20]
plot_images(test_X[p].reshape(-1, 28, 28), predictions[p])

imgBase0 = imageio.imread('data/0.png', as_gray=1)
img = imgBase0/255.0
dat = np.reshape(img, 784)

pre = mlp.predict(dat)
plot_images(img, [0])



