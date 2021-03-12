import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

mnist_train = pd.read_csv('mnist_train.csv', header = None)
mnist_test = pd.read_csv('mnist_test.csv', header = None)

# print(mnist_train.shape)

cols = ['label']
for i in range(784):
    cols.append('px_{}'.format(i+1))

# print(cols)

mnist_train.columns = cols
mnist_test.columns = cols

# print(mnist_train.head(5))

image_row = mnist_train.values[10, 1:]
image = image_row.reshape(28, 28)

# plt.imshow(image, cmap= 'Greys') TODO Picture
# plt.show()

from sklearn.neighbors import KNeighborsClassifier

train_data = mnist_train.values[:, 1:]
test_data = mnist_test.values[:, 1:]

train_label = mnist_train.values[:, 0]
test_label = mnist_test.values[:, 0]

print(train_data.shape, test_data.shape)
print(train_label.shape, test_label.shape)

kn_classifier = KNeighborsClassifier(n_jobs= -1) # Количество параллелльных комманд

kn_classifier = kn_classifier.fit(train_data, train_label)

test_id = 10 # ID числа из БД
plt.imshow(test_data[test_id, :].reshape(28, 28), cmap = 'Greys')
plt.show()

print('На рисунке цифра {}'.format(test_label[test_id]))

kn_prediction = kn_classifier.predict(test_data)
print('Accuracy: {}%'.format(accuracy_score(test_label, kn_prediction) * 100))

print(accuracy_score(test_label, kn_prediction) *1000)


print(kn_prediction)
print(kn_classifier.predict(test_data[test_id, :].reshape(1, 784)))

mlp_classifier = MLPClassifier(verbose= True)
mlp_classifier = mlp_classifier.fit(train_data, train_label)


test_id = 10 # ID числа из БД
plt.imshow(test_data[test_id, :].reshape(28, 28), cmap = 'Greys')
plt.show()

mlp_classifier.predict(test_data[test_id, :].reshape(1, 784))
mlp_prediction = mlp_classifier.predict(test_data)
print('Accuracy: {}%'.format(accuracy_score(test_label, mlp_prediction) * 100))


