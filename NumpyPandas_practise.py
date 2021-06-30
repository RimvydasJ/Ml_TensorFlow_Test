import numpy as np
import pandas as pd

one_dimensional_array = np.array([1.2,2.4,3.5,4.7,6.1,7.2,8.3,9.5])
#print(one_dimensional_array)

two_dimenesional_aray = np.array([[6,5],[11,7],[4,8]])
#print(two_dimenesional_aray)


#print(np.zeros((2,3))) #2 rows 3 columns

sequence_of_integers = np.arange(2,12)
#print(sequence_of_integers)

random_integers = np.random.randint(low=50, high=101, size=6)
#print(random_integers)


#print(np.random.random([6])) #Random between 0.0 and 1.0

feature = np.arange(5,21)
label = ((feature * 3) + 4)
noise = np.random.uniform(low=-2, high=2, size=16)

label = label + noise

#print(feature)
#print(label)

my_data = np.array([[0,3],[10,7],[20,9],[30,14],[40,15]])
my_column_names = ['temperature','activity']
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

#print(my_dataframe)

my_dataframe["adjusted"] = my_dataframe["activity"] + 2
#print(my_dataframe)

print(my_dataframe.head(2))
print(my_dataframe.iloc[[2]])

data = np.random.randint(low=-1, high=101, size=(3,4))
names = ["Eleanor", "Chidi", "Tahani", "Jason"]
dataframe = pd.DataFrame(data=data,columns=names)
dataframe["Janet"] = dataframe["Tahani"] + dataframe["Jason"]
print(dataframe)
