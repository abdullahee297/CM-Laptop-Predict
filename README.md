🎓 Laptop Eligibility Prediction Using Machine Learning

This project uses Logistic Regression to predict whether a university student is eligible to receive a laptop based on their Department, HSC (HEC) Percentage, and CGPA.

The dataset may contain missing student data, so those entries are removed to ensure accurate predictions.

📌 Project Objective

Predict laptop eligibility (Yes / No) for students
Handle missing student data
Encode categorical features properly
Train and evaluate a Logistic Regression model
Visualize model performanceRR
Allow real-time user input for prediction

🧠 Technologies & Libraries Used

Python
Pandas – Data handling
Scikit-learn – Machine learning
Matplotlib – Visualization

📂 Dataset Description (laptop.csv)

The dataset contains the following columns:

Column Name	Description
Department	Student’s department (CS, IT, EE, etc.)
HSC Percentage	Higher Secondary Certificate percentage
CGPA	Cumulative Grade Point Average
Status	Laptop eligibility (Yes / No)
🛠 Step-by-Step Explanation
1️⃣ Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


🔹 These libraries help in:

Reading data
Encoding categorical values
Training the ML model
Evaluating accuracy
Visualizing results

2️⃣ Load the Dataset
lap = pd.read_csv("laptop.csv")

🔹 Reads the dataset from a CSV file into a Pandas DataFrame.

3️⃣ Data Cleaning & Preprocessing
lap["Status"] = lap["Status"].str.strip()
lap = lap.dropna()


✔ Removes:

Extra spaces in the Status column
Any student record with missing data

📌 This ensures only complete student data is used for training.

4️⃣ Encode Categorical Data
lap["Department"] = dep_encoder.fit_transform(lap["Department"])
y = status.fit_transform(lap["Status"])

🔹 Machine learning models cannot understand text, so:

Departments are converted into numbers
Status (Yes / No) is converted into 0 / 1

5️⃣ Feature Selection
x = lap[["Department", "HSC Percentage", "CGPA"]]


🔹 These features are used to predict laptop eligibility.

6️⃣ Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)


✔ 80% data → Training
✔ 20% data → Testing

This helps test how well the model performs on unseen data.

7️⃣ Train the Logistic Regression Model
model = LogisticRegression()
model.fit(x_train, y_train)


🔹 Logistic Regression is ideal for binary classification problems like:

Eligible / Not Eligible

8️⃣ Model Prediction & Evaluation
🔹 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)


Shows:

True Positives
True Negatives
False Positives
False Negatives

🔹 Accuracy Score
acc = accuracy_score(y_test, y_pred)

📊 Displays how accurately the model predicts eligibility.

9️⃣ Data Visualization
🔹 Accuracy Bar Chart

Shows overall model performance.

🔹 Scatter Plot
plt.scatter(lap["HSC Percentage"], lap["CGPA"], c=y)


📈 Visualizes how CGPA and HSC Percentage affect laptop eligibility.

🔟 User Input Prediction
dep = input("Enter the Department: ")
hec = float(input("Enter your HEC Percentage: "))
cgpa = float(input("Enter your CGPA: "))


🔹 Takes real-time student data from the user.

result = status.inverse_transform(pred_data)[0]
print("Result for selection:", result)


✔ Outputs whether the student is Eligible or Not Eligible for the laptop.

📊 Output Example
Enter the Department: CS
Enter your HEC Percentage: 85
Enter your CGPA: 3.5

Result for selection: Yes

🚀 Future Improvements

Add real university dataset
Improve accuracy with feature scaling
Try advanced models (SVM, Random Forest)
Create a web or GUI interface

📌 Conclusion

This project demonstrates how Machine Learning can be used to:
Automate decision-making
Handle missing data
Predict student eligibility fairly
It’s ideal for students learning ML, logistic regression, and real-world data preprocessing.
