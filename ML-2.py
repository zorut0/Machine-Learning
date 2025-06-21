import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# a) Create a Series using pandas and display
data = pd.Series([1, 2, 3, 4, 5])
print(data)

# b) Access the index and the values of our Series
print(" Index:", data.index)
print("Values:", data.values)

# c) Compare an array using Numpy with a series using pandas
array = np.array([1, 2, 3, 4, 5])
print(array)
comparison = data.equals(pd.Series(array))
print(" Are they equal?", comparison)

# d) Define Series objects with individual indices
data_with_index = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(data_with_index)

# e) Access single value of a series
single_value = data[2]
print("Single value:", single_value)

# f) Load datasets in a Data frame variable using pandas
df = pd.read_csv("CarPrice.csv")
print(df.head())

# g) Usage of different methods in Matplotlib
plt.plot(data)
plt.title('Sample Plot')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()
