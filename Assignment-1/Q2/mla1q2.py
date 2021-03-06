# -*- coding: utf-8 -*-
"""MLA1Q2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HfhFNtl_VSLLpOmzux_P7MIdLPwrI6li
"""

import pandas as pd
import scipy.io
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier as dtc

from google.colab import drive
drive.mount('/content/drive')

D = scipy.io.loadmat('/content/drive/My Drive/ML(PG)_assignment_1/dataset_2.mat')
print(D)

samplesarr2 = D['samples']
data2 = pd.DataFrame(samplesarr2)
labels2 = D['labels'].reshape(-1)
data2['labels'] = labels2
print(data2)

"""# **Answer to Q2**"""

data2.head(10)

def Train_Test_Split(df):
  #shuffle_df = df.sample(frac=1)
  # Define a size for your train set 
  train_size = int(0.7 * len(df))

  # Split your dataset 
  train_set = df[:train_size]
  test_set = df[train_size:]
  return train_set,test_set

X_train,X_test=Train_Test_Split(data2.iloc[:,:2])

y_train,y_test=Train_Test_Split(data2.iloc[:,2:])

cls=dtc(max_depth=3,class_weight='balanced')

cls.fit(X_train,y_train)

y_pred=cls.predict(X_test)

y_test1=np.array(y_test)

y_test1 = (y_test1).reshape(-1)
y_test1

df_confusion = pd.crosstab(y_test1, y_pred)
df_confusion

t_val=0
f_val=0
for i in range(4):
  for j in range(4):
    if i==j:
      t_val+=df_confusion[i][j]
    else:
      f_val+=df_confusion[i][j]

accuracy=t_val/(f_val+t_val)

accuracy

"""# **Answer to Q2(a)**"""

acc_array1=[]
def Grid_Search():
  for i in range(2,17):
    cls=dtc(max_depth=i,class_weight='balanced')
    cls= cls.fit(X_train,y_train)
    y_pred=cls.predict(X_test)
    y_test2=np.array(y_test)
    y_test2 = (y_test2).reshape(-1)
    df_confusion2 = pd.crosstab(y_test2, y_pred)
    t_val1=0
    f_val1=0
    for i in range(4):
      for j in range(4):
        if i==j:
          t_val1+=df_confusion2[i][j]
        else:
          f_val1+=df_confusion2[i][j]
    accuracy1=t_val1/(f_val1+t_val1)
    acc_array1.append(accuracy1)
  return acc_array1

acc_array1 = Grid_Search()
plt.plot(np.arange(2,17), acc_array1, marker = 'o')
plt.show()

#For Depth vs Validation_accuracy table
table1=np.vstack((np.arange(2,17),acc_array1)).T
table1_df=pd.DataFrame(table1,columns=["Depth","Validation_Accuracy"])
table1_df

"""# **Answer to Q2(b)**"""

acc_array2=[]
def Grid_Search2():
  for i in range(2,17):
    cls=dtc(max_depth=i,class_weight='balanced')
    cls= cls.fit(X_train,y_train)
    y_pred2=cls.predict(X_train)
    y_train2=np.array(y_train)
    y_train2 = (y_train2).reshape(-1)
    df_confusion3 = pd.crosstab(y_train2, y_pred2)
    t_val2=0
    f_val2=0
    for i in range(4):
      for j in range(4):
        if i==j:
          t_val2+=df_confusion3[i][j]
        else:
          f_val2+=df_confusion3[i][j]
    accuracy2=t_val2/(f_val2+t_val2)
    acc_array2.append(accuracy2)
  return acc_array2

acc_array2 = Grid_Search2()
plt.plot(np.arange(2,17), acc_array2, marker = 'o')
plt.show()

train_acc = pd.DataFrame(acc_array2)
train_acc
table2_df = table1_df
table2_df['Training_accuracy'] = train_acc
table2_df

"""# **Answer to Q2(c)**"""

from sklearn import metrics
acc_array3=[]
def Grid_Search3():
  for i in range(2,17):
    cls=dtc(max_depth=i,class_weight='balanced')
    cls= cls.fit(X_train,y_train)
    y_pred=cls.predict(X_test)
    y_test2=np.array(y_test)
    y_test2 = (y_test2).reshape(-1)
    accuracy3=metrics.accuracy_score(y_test,y_pred)
    acc_array3.append(accuracy3)
  return acc_array3

acc_array3 = Grid_Search3()
plt.plot(np.arange(2,17), acc_array3, marker = 'o')
plt.show()

acc_array1

valid_acc = pd.DataFrame(acc_array3)
table3_df = table1_df.iloc[:,:2]
table3_df['Validation_Accuracy(metrics)'] = valid_acc
acc_array1  = Grid_Search()
diff_arr = np.subtract(acc_array1, acc_array3)
diff = pd.DataFrame(diff_arr)
table3_df['Difference'] = diff
table3_df