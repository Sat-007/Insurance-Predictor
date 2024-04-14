
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#C:/Users/sathv/OneDrive/Desktop/Insurance_predictor/
insurance_dataset = pd.read_csv("insurance.csv")
#print(insurance_dataset)


#EDA
insurance_dataset.info()

#hence we have found categorical featrures of our dataset which are sex, smoker, region


#insurance_dataset.sum()

sns.set()
sns.displot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()