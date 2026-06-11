import pandas as pd

winedatac=pd.read_csv('D:/Mechine Learning 4063/wine.csv')

print(winedatac)
print(winedatac.head())
print("shape\n",winedatac.shape)
print("columns\n",winedatac.columns)
print("dtypes\n",winedatac.dtypes)
print("ndim\n",winedatac.ndim)
print("size\n",winedatac.size)


winedatae=pd.read_excel('D:/Mechine Learning 4063/wine.xls')

print('\n')
