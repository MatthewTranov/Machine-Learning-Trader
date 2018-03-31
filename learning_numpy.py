import numpy as np
from scipy.stats import bernoulli
from deap import base, creator, tools, algorithms

Z = np.array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
print(Z.sum(0))
print(Z.sum(1))

Z = np.array([[[0],  [1], [2],  [3]],
              [[4],  [5], [6],  [7]],
              [[8],  [9], [10], [11]]])
print(Z.sum(0))
print(Z.sum(1))
print(Z.sum(2))

X = np.empty((1,4))
Y = np.empty((1))
print(X)
print(Y)

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(data[1, 0:])


data = np.empty(0)
data = np.append(data,  np.array([1, 2, 3]))
data = np.append(data, np.array([4, 5, 6]))

print(data)
data = np.reshape(data, (len(data), 1))

print(data)

print(bernoulli.rvs(0.4, size=1000))

toolbox = base.Toolbox()

list1 = [0, 1, 2, 3]
list2 = [0, 3, 2, 1]
print(tools.cxOrdered(list1, list2))
print(list1, list2)

print('--------CREATION AND POPULATION OF EMPTY ARRAYS-----------')

print('''data = np.empty(0) 
data = np.append(data,  np.array([1, 2, 3]))
data = np.append(data, np.array([4, 5, 6]))''')
data = np.empty(0)
print(data)
data = np.append(data,  np.array([1, 2, 3]))
print(data)
data = np.append(data, np.array([4, 5, 6]))
print(data)

print('''data2 = np.empty(0,3) 
data2 = np.vstack(data2,  np.array([1, 2, 3]))
data2 = np.hstack((data2,  np.array([1, 2, 3])))
data2 = np.vstack(data2, np.array([4, 5, 6]))
data2 = np.vstack((data2, np.array([6, 7, 8])))
for element in data2:
    print(element)''')
data2 = np.empty((0, 3))
print(data2)
data2 = np.vstack((data2,  np.array([1, 2, 3])))
print(data2)
data2 = np.vstack((data2, np.array([4, 5, 6])))
print(data2)
data2 = np.vstack((data2, np.array([6, 7, 8])))
print(data2)
for element in data2:
    print(element)
array_list = []
array_list.append(data2)
array_list.append(data2)
print(array_list)

print('''data3 = np.empty(0,3) 
data = np.vstack(data3,  data2)''')
data3 = np.empty((0, 4))
print(data3)
#data3 = np.vstack(data3,  data2)
print(data3)

for i in range(1,10):
    print(i)

WINDOW_SIZE = 3
price_data = [[1,2,3,4,5,6,7,8],[11,22,33,44,55,66,77,88]]
x_data = []
y_data = []
for stock in price_data:
    # go through each window of prices not counting the last one which
    # can't be used for prediction
    price_tuples_list = []
    next_change_list = []
    for i in range(len(stock) - WINDOW_SIZE - 1):
        # add the window to the list
        price_tuples_list.append(stock[i:i+3])
        # add the next price to the list
        next_change_list.append(stock[i+3])
    x_data.append(np.asarray(price_tuples_list))
    y_data.append(np.asarray(next_change_list))
print(x_data)
print(y_data)