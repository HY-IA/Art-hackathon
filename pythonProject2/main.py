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

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5, stratify = Y, random_state=5)

model = LogisticRegression(max_iter=100000)


model.fit(X,Y)

x= input("whats your sr?")
x2= input("whats your rr?")
x3= input("whats your t?")
x4= input("whats your lm?")
x5= input("whats your bo?")
x6= input("whats your rem?")
x7= input("whats your sr.1?")
x8= input("whats your hr?")
file = open("test.csv", "a")
file.write(x+","+x2+","+x3+","+x4+","+x5+","+x6+","+x7+","+x8)

file.close()
input1 = pd.read_csv("test.csv")
print(input1)
X_prediction = model.predict(input1)


print(X_prediction)










