pip install matplotlib
pip install scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

# Define the Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Define the KNN class
class KNN:
    def __init__(self, k=10):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        distances = [euclidean_distance(x, X_train) for X_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common

# Load the dataset
df = pd.read_csv('SIRTUIN6.csv')

# Encode the 'Class' column
df = pd.get_dummies(df, columns=['Class'], drop_first=True)

# Split the dataset into features and target variable
X = df[['SP-6', 'FMF', 'SC-5', 'SHBd', 'minHaaCH', 'maxwHBa']]
y = df['Class_Low_BFE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Create and fit the KNN classifier
clf = KNN(k=5)
clf.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
test_predictions = clf.predict(X_test)
accuracy = np.mean(test_predictions == y_test)

# Define the mapping
label_mapping = {0: 'High_BFE', 1: 'Low_BFE'}

# Streamlit app
st.title('KNN Classifier for SIRTUIN6 Data')
st.write('Input the features to get the prediction:')

# Input features
SP_6 = st.number_input('SP-6', value=0.0)
FMF = st.number_input('FMF', value=0.0)
SC_5 = st.number_input('SC-5', value=0.0)
SHBd = st.number_input('SHBd', value=0.0)
minHaaCH = st.number_input('minHaaCH', value=0.0)
maxwHBa = st.number_input('maxwHBa', value=0.0)

# Prediction button
if st.button('Predict'):
    new_data = np.array([[SP_6, FMF, SC_5, SHBd, minHaaCH, maxwHBa]])
    prediction = clf.predict(new_data)[0]
    mapped_prediction = label_mapping[prediction]
    st.write(f'The prediction for the input data is: **{mapped_prediction}**')

# Display accuracy (learning rate)
st.write(f'Accuracy of the model (learning rate): {accuracy:.2f}')

# Scatter plot parameters selection
st.write('Scatter plot selection:')
features = ['SP-6', 'FMF', 'SC-5', 'SHBd', 'minHaaCH', 'maxwHBa']
x_axis = st.selectbox('Select X-axis', features)
y_axis = st.selectbox('Select Y-axis', features)

# Plotting section
st.write(f'Scatter plot of {x_axis} vs {y_axis}:')
cmap = ListedColormap(['blue', 'orange'])
fig, ax = plt.subplots()
scatter = ax.scatter(X[x_axis], X[y_axis], c=y, cmap=cmap, edgecolor='k', s=20)
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title(f'Scatter Plot of {x_axis} vs {y_axis}')
# Adding legend
legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax.add_artist(legend1)
st.pyplot(fig)
