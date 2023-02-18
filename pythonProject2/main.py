# this is a project for the ART-HACK 2023
# done by : Mo and Hiro

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sleep_data = pd.read_csv("SaYoPillow.csv")

low_normal = sleep_data[sleep_data.sl == 0]
medium_low = sleep_data[sleep_data.sl == 1]
medium = sleep_data[sleep_data.sl == 2]
medium_high = sleep_data[sleep_data.sl == 3]
high = sleep_data[sleep_data.sl == 4]

X = sleep_data.drop(columns="sl", axis = 0)
Y = sleep_data["sl"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, stratify = Y, random_state=5)

model = LogisticRegression()
model.max_iter=1000

model.fit(X_train,Y_train)
f = X_test.
X_prediction = model.predict(X_test)
training_accuracy = accuracy_score(X_prediction,Y_test)
print(X_test)
print(X_prediction)
print(Y_test)





