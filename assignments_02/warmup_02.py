import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- scikit-learn API ---
#scikit-learn Q1
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
new_years = np.array([4, 8]).reshape(-1, 1)

#Create
model = LinearRegression()

#Fit
model.fit(years, salary)

#Predict
predicted = model.predict(new_years)

print(f'slope: {model.coef_[0]}')
print(f'intercept: {model.intercept_}')
print(f'prediction 1: {predicted}')

#scikit-learn Q2
x = np.array([10, 20, 30, 40, 50])
print(f'shape before conversion: {x.shape}')
#the reshape method does not change the original array
x = x.reshape(-1, 1) 
print(f'shape after conversion: {x.shape}')

#Q: why scikit-learn needs X to be 2D?
#A: It relies on the consistent matrix structure (num_samples, num_features)

#scikit-learn Q3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
print(X_clusters.shape)
#Create
kmeans = KMeans(n_clusters=3, random_state=42)
#Fit
kmeans.fit(X_clusters)
#Predict
labels = kmeans.predict(X_clusters)
centers = kmeans.cluster_centers_
print(f'Cluster centers: {centers}')
print(f'# of points in each cluster: {np.bincount(labels)}')

#Then create a scatter plot coloring each point by its cluster label, plot the cluster centers as black X's, add a title and axis labels. Save the figure to outputs/kmeans_clusters.png.

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c = labels, cmap='viridis', s=60, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', 
           s=100, label='Centroids')
plt.title("Student Clusters Found by K-Means")
plt.xlabel("Study Hours (synthetic scale)")
plt.ylabel("Social Time (synthetic scale)")
plt.savefig('./outputs/kmeans_clusters.png')
plt.close()

# --- Linear Regression ---
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

#Linear Regression Q1
plt.scatter(age, cost, c=smoker, cmap='coolwarm', s=60, alpha=0.7)
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")
plt.savefig('./outputs/cost_vs_age.png')
plt.close()
#Q: Are there two distinct groups visible? What does that suggest about the smoker variable?

#A: There are 2 distinct groups visible.  The group with smoker status has higher medical cost.


#Linear Regression Q2
age = age.reshape(-1, 1)
#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    age, cost, test_size=0.2, random_state=42
)
print(f'age train shape: {X_train.shape}')
print(f'age test shape: {X_test.shape}')
print(f'cost train shape: {y_train.shape}')
print(f'cost test shape: {y_test.shape}')

#Linear Regression Q3
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

#rmse and r2
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R²:", r2)

#Add a comment interpreting the slope in plain English -- what does it mean for medical costs?
#Slope: 196.575
#It tells me how much the medical cost increases per year of age without knowing the smoker status.


#Linear Regression Q4
X_full = np.column_stack([age, smoker])
X_train2, X_test2, y_train_2, y_test_2 = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)
model_full = LinearRegression()
model_full.fit(X_train2, y_train_2)
y_pred2 = model_full.predict(X_test2)
rmse = np.sqrt(mean_squared_error(y_test_2, y_pred2))
r2 = r2_score(y_test_2, y_pred2)
print("RMSE:", rmse)
print("R²:", r2)
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
#Q: Compare it to the R² from Question 3 -- does adding the smoker flag help?
#A: Adding the smoker flag helps improve the r2.

#Q: Add a comment interpreting the smoker coefficient: what does it represent in practical terms?
#A: The smoker coefficient of 14538.03 means the annual medical cost is $14538.03 higher compared to the non-smoker of the same age.  

#Linear Regression Q5
# Using the two-feature model from Linear Regression Question 4, create this plot for the test set. Add a diagonal reference line, a title "Predicted vs Actual", labeled axes, and save to outputs/predicted_vs_actual.png.

#Model predictions, y_pred2, go on x-axis
#True value: y_test_2 go on y-axis
min_val = min(y_pred2.min(), y_test_2.min())
max_val = max(y_pred2.max(), y_test_2.max())

plt.scatter(y_pred2, y_test_2, color='blue', alpha=0.5, label='Observations')
plt.plot([min_val, max_val], [min_val , max_val], color='red', label='Reference line',)
plt.xlabel("Predicted value")
plt.ylabel("True value")
plt.title("Predicted vs Actual")
plt.legend()
plt.savefig('./outputs/predicted_vs_actual.png')

#Q: what does it mean when a point falls above the diagonal? What about below?
#A: When a point falls above the diagonal, it means the true value is higher than the predicted value; the model underestimated the cost.  When a point falls below the diagonal, it means the true value is lower than the actual value; the model overestimated the cost.
