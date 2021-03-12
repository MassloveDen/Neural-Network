import numpy as np
import pandas as pd
my = [1,2,3,4,5]
type(my)
list
numpy = np.array(my)
type(numpy)
# numpy.ndarray
# mat = np.array([
#     [1, 2, 3],
#     [4, 5, 6]
# ])
# print(mat)

test_data = pd.read_csv('mnist_test.csv',header= None)
image_row = test_data.values[3, 1:]

image_matrix = image_row.reshape(28, 28)
import matplotlib.pyplot as plt

plt.imshow(image_matrix, cmap= 'Greys')
plt.show()

