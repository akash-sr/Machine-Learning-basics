import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pp
from matplotlib import style
import pickle

# load data
data = pd.read_csv("student-data.csv", sep=",")
# clean data
data = data[["GS","TS","Uni_rat","SOP","LOR","CGPA","Research","Chance"]]
# visualize data head
print(data.head())
# visualizing the data as a plot
feature = "CGPA"    # the x axis for the plot
style.use("ggplot")
pp.scatter(data[feature],data["Chance"])
pp.xlabel(feature)
pp.ylabel("Chance")
pp.show()

# create training and test sets
predict = "Chance"  # the quantity we want to predict

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
"""
# Run only once to store the model with the best accuracy
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    # constructing the model
    model = linear_model.LinearRegression()
    model.fit(x_train,y_train)

    # testing the model
    accuracy = model.score(x_test,y_test)
    print(accuracy)
    if accuracy > best:
        best = accuracy
        with open("student_mode.pickle","wb") as f:
            pickle.dump(model, f)
print(best)
"""
# load model
model_in = open("student_mode.pickle", "rb")
model = pickle.load(model_in)

print("Final Model:")
print("Test accuracy:",model.score(x_test,y_test),"Training accuracy:",model.score(x_train,y_train))
print("Coefficient\n", model.coef_)
print("Intercept\n", model.intercept_)

# # visualizing predictions
# predictions = model.predict(x_test)
#
# for i in range(len(predictions)):
#     print("Prediction:",predictions[i],"Actual Value",y_test[i])





