# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Data Preparation: Collect and preprocess energy data, ensuring it is clean, standardized, and ready for PCA application.
Covariance Computation: Compute the covariance matrix of the data to identify feature correlations.
Principal Component Selection: Calculate eigenvalues and eigenvectors, selecting the top components that explain the most variance.
Dimensionality Reduction: Transform the data into the new feature space defined by the selected principal components.

## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: Priyadharshan S
RegisterNumber:  212223240127
*/
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import accumulate

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from scipy.stats import loguniform

hwdf = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/HeightsWeights.csv', index_col=0)
hwdf.head()

scaler = StandardScaler()
hwdf[:] = scaler.fit_transform(hwdf)
hwdf.columns = [f'{c} (scaled)' for c in hwdf.columns]
hwdf.head()

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
xs, ys, zs = [hwdf[attr] for attr in hwdf.columns]
ax1.scatter(xs, ys, zs)

ax2 = fig.add_subplot(122, projection='3d')
xs, ys, zs = [hwdf[attr] for attr in hwdf.columns]
ax2.view_init(elev=10, azim=-10)
ax2.scatter(xs, ys, zs)

plt.tight_layout()
plt.show()

sns.pairplot(hwdf)
plt.show()

pca = PCA()
pca.fit(hwdf)

Xhat = pca.transform(hwdf)
Xhat.shape
```

## Output:
![image](https://github.com/user-attachments/assets/2df7691a-4a60-4627-a22c-a1e2ab2837b4)
![image](https://github.com/user-attachments/assets/b3cc5206-5f1f-4f4b-ba14-0bbad8d3ca72)
![image](https://github.com/user-attachments/assets/1f8f4d3f-0ee6-4b2c-855e-60e424c4157d)

## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
