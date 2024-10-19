import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import numpy as np

#reading the csv files using pandas
train_df = pd.read_csv('hearttrain.csv')

test_df = pd.read_csv('hearttest.csv')

#prints the datatypes of the features
print(train_df.info())

#printd the mean,median,maximum value,minimum values for every feature
print(train_df.describe())

#finding the missing values in the data
print(train_df.isnull())

#prints the total null values for every feature
print(train_df.isnull().sum())

#prints the first five rows of the data
print(train_df.head())

#prints last 5 row of the data
print(train_df.tail())

#removing the index column axis=1 - removing total column
list1=['index']
train_df.drop(list1,axis=1,inplace=True)

#heatmap between the features and the target value -i.e correlation co-efficient for every feature corressponding to the target value
plt.figure(dpi=125)
sns.heatmap(np.round(train_df.corr(numeric_only=True),2),annot=True)
plt.show()

# Step 2: Preprocess the data# Separate features and target
X = train_df[[ 'resting bp s', 'cholesterol', 'fasting blood sugar','exercise angina','oldpeak']]
y = train_df['target']

X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#model selection-using entropy to select the root node
model =DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=5,min_samples_leaf=5)
#training the data
model.fit(X_train, y_train)
# Evaluate the model on validation set
y_pred = model.predict(X_test)

#scatter plot  between the predicted values and actual values
plt.scatter(y_pred,Y_test)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("predicted values vs actual values")
plt.show()

#ggraph between the max heart rate and target value
plt.scatter(train_df['max heart rate'],train_df['target'])
plt.xlabel("max heart rate")
plt.ylabel("target")
plt.title(" max heart rate vs target")
plt.show()

#decision tree in text format
text_representation=tree.export_text(model)
print(text_representation)

#decision tree representation using nodes
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=[ 'resting bp s', 'cholesterol', 'fasting blood sugar','exercise angina','oldpeak'], class_names=['target0', 'target1'])
plt.show()

acc=accuracy_score(Y_test,y_pred)
print(acc)
report=classification_report(Y_test,y_pred)
print(f" classification reporrt:")
print(report)

