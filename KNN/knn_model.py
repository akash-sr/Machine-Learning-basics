import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data", sep=",")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

X = list(zip(buying, maint, doors, persons, lug_boot, safety))
Y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)
predictions = model.predict(x_test)
classes = ["unacc", "acc", "good", "vgood"]
for i in range(len(predictions)):
    print("Actual:",classes[y_test[i]],"Prediction:",classes[predictions[i]])
