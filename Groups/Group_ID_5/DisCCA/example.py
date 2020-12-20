'''
Program containing example use of DisCCA
'''
import DisCCA
import pandas as pd

first_dataframe=pd.read_csv("first_view",delim_whitespace=True,header=None)
second_dataframe=pd.read_csv("second_view",delim_whitespace=True,header=None)



target=[]
for i in range(10):
    for j in range(200):
        target+=[i]

D_CCA=DisCCA.DisCCA()
D_CCA.fit(first_dataframe,second_dataframe,target, 47)
print(first_dataframe.shape,second_dataframe.shape)
transformed_first_dataframe,transformed_first_dataframe=D_CCA.transform(first_dataframe, second_dataframe)

