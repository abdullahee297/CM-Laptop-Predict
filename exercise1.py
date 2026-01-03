#you have given a dataset of a university in which you have to tell whether the studend 
# with his hec percentage, cgpa and department can get the laptop may be there are some data missing in the set
# so you have to remove that student

# 1 import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# 2 import the dataset
lap = pd.read_csv("laptop.csv")

model = LogisticRegression()
dep_encoder = LabelEncoder()
status = LabelEncoder()
sc = StandardScaler()

lap["Status"] = lap["Status"].str.strip()

#lap.info()
#print(lap.isnull().values.any())
#checked no nan value if then this will remove that student
lap = lap.dropna()

lap["Department"] = dep_encoder.fit_transform(lap["Department"])

# 3 split the data into train test

x = lap[["Department", "HSC Percentage", "CGPA"]]
y = status.fit_transform(lap["Status"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 4 Training the logistic regression

model.fit(x_train, y_train)

from sklearn.metrics import ConfusionMatrixDisplay

# Predictions
y_pred = model.predict(x_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=status.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

plt.bar(["Accuracy"], [acc], color="green")
plt.ylim(0, 1)
plt.title("Model Accuracy")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(lap["HSC Percentage"], lap["CGPA"], c=y, cmap="bwr", edgecolor="k")
plt.xlabel("HSC Percentage")
plt.ylabel("CGPA")
plt.title("Laptop Eligibility by Status")
plt.colorbar(label="Status (0/1)")
plt.show()


# 5 Relavant data to predict

dep = input("Enter the Deparment: ")
hec = float(input("Enter you HEC Percentage: "))
cgpa = float(input("Enter your CGPA: "))

dep_num = dep_encoder.transform([dep])[0]

get_data = [[dep_num, hec, cgpa]]
pred_data = model.predict(get_data)

# 6 Results

result = status.inverse_transform(pred_data)[0]
print("Result for selection: ", result)