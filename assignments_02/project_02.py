import os
import numpy as np
import pandas as pd
#To resolve the downcasting behavior deprecating warning
pd.set_option('future.no_silent_downcasting', True)
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

#student_performance_math.csv
#in pd.read_csv(), I will need to specify the sep value that each field is separated by a semicolon, instead of a comma.

# Task 1: Load and Explore
df = pd.read_csv('./student_performance_math.csv', sep=';')
print(df.shape)
print(df[:5])
print(df.dtypes)

#G3
plt.hist(df['G3'], bins=21, color='green', edgecolor='black')
plt.xlabel('Final Math Grades')
plt.ylabel('Count')
plt.title('Distribution of Final Math Grades')
plt.savefig('./outputs/g3_distribution.png')
plt.close()

# Task 2: Preprocess the Data
df_clean = df[df['G3'] > 0]
print(f'Shape before the filter: {df.shape}')
print(f'Shape after the filter: {df_clean.shape}')

#Q: why would keeping these G3 = 0 rows distort the model?
#A: They are outliers and can disrupt the distribution.

#convert the yes/no columns to 1/0 and sex column to 0/1
to_convert = {
  'yes' : 1,
  'no' : 0,
  'M' : 0,
  'F' : 1
}
df_clean = df_clean.replace(to_convert)
print(df_clean)

#Pearson correlation between absences and G3 on both the original dataset and the filtered one, and print both values. The difference is striking. 
print(df[['absences', 'G3']].corr())
print(df_clean[['absences', 'G3']].corr())

plt.scatter(df['absences'], df['G3'], color='red')
plt.xlabel('absences')
plt.ylabel('G3')
plt.show()

plt.scatter(df_clean['absences'], df_clean['G3'], color='blue')
plt.xlabel('absences')
plt.ylabel('G3')
plt.show()

#Q: What were students with G3=0 doing in the original data that made absences look like a weak predictor? You might want to explore scatter plots to help understand this.

#A: Students with G3=0 were outliers which brought the correlation between absences and G3 score to 0.  After filtering them out, the actual correlation is negative.  As the absences increase, the G3 score decreases.


# Task 3: Exploratory Data Analysis
corrs = df_clean.corr()
corrs_g3 = corrs['G3']
print(corrs_g3.sort_values())
#Q: Which feature has the strongest relationship with G3? Are any results surprising?
#A: Number of past class failures has the strongest relationship with G3.  I am surprised that the extra educational support from the school has negative effect on the G3 grade.

#Failures vs G3 boxplot
sns.boxplot(x='failures', y='G3', data=df_clean)
plt.title("Number of Failures vs G3")
plt.xlabel("Number of Failures")
plt.ylabel("G3")
plt.savefig('outputs/failures_vs_g3_boxplot.png')
plt.close()
#As the number of past failure increases, the median of G3 decreases, showing a strong negative relationship.

#Weekly study time vs G3 scatter plot
sns.boxplot(x='studytime', y='G3', data=df_clean)
plt.title("Weekly study time vs G3")
plt.xlabel("Weekly study time")
plt.ylabel("G3")
plt.savefig('outputs/studytime_vs_g3_boxplot.png')
plt.close()
#As the study time increases, G3 increases.  However, this is not as strong as number of past failure.

# Task 4: Baseline Model

#Build the simplest possible model: use failures alone to predict G3. Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, and print the slope, RMSE, and R² on the test set.
failures_x = df_clean[["failures"]] #2D
G3_y = df_clean["G3"]

#train-test split
failures_x_train, failures_x_test, G3_y_train, G3_y_test = train_test_split(
  failures_x, G3_y, test_size=0.2, random_state=42
)

#model fit
model = LinearRegression()
model.fit(failures_x_train, G3_y_train)
G3_y_pred = model.predict(failures_x_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

rmse = np.sqrt(mean_squared_error(G3_y_test, G3_y_pred))
r2 = r2_score(G3_y_test, G3_y_pred)

print("RMSE:", rmse)
print("R²:", r2)

#Q: Given that grades are on a 0-20 scale, what do the slopes and RMSE tell you in plain English? Is R² better or worse than you expected from exploratory data analysis?

#A: For each additional past failure, G3 grade will be lowered by 1.43 points.  The prediction of G3 will be off by 2.96 points.  R² is worse than I expected since the correlation between failures and G3 was -0.29.


# Task 5: Build the Full Model
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
x = df_clean[feature_cols].values
y = df_clean["G3"].values

#split
x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.2, random_state=42
)

#model fit
model = LinearRegression()
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)

y_train_pred = model.predict(x_train)

#print both train R² and test R², as well as RMSE on the test set. Compare the test R² to your baseline from Task 4 -- 
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)

print("test RMSE:", rmse_test)
print("test R²:", r2_test)
print("train R²:", r2_train)

#Q: how much does adding more features help?
#A: Adding more features helps by improving R2 from 0.089 to 0.154.

#Coefficient of each feature
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

#Look carefully at the coefficients. Sort them mentally from largest to smallest. 

#Q: Are any signs (positive or negative) surprising given what you know about the data? For any surprising result, add a comment with your best explanation. 
#A: Extra educational support from the school is surprising because it worsens the G3 grade rather than improves it.  
 
#Q: Then compare train R² to test R² -- are they close, or is there a gap? What does that tell you about the model?
#A: They are close which confirms that the model is stable.

#Q: if you were deploying this model in production, which features would you keep and which would you drop? Justify your choices based on what you see in the numbers.
#A: I would keep failures, internet, higher, and studytime as they all have strong coefficient.  I would drop freetime, activities, and medu as they have close to 0 coefficient.  I will also drop schoolsup as it is giving the opposite of the expected result making the model less useful.

# Task 6: Evaluate and Summarize
# predicted vs actual plot
plt.scatter(y_test_pred, y_test, color='blue', alpha=0.5, label='Observations')
plt.plot([0, 20], [0, 20], color='red', label='Reference line',)
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.title("Predicted vs Actual (Full Model)")
plt.legend()
plt.savefig('./outputs/predicted_vs_actual.png')
plt.close()


#Q: does the model seem to struggle more at the high end, the low end, or is error roughly uniform across grade levels? What does a value above or below the diagonal mean?

#A: It seems the error is roughly uniform across grade levels.  A value above the diagonal line means the model underestimates it and a value below means the model overestimates it.

#Q: Summary
#A: The size of the filtered dataset is 357 and the size of the testset is 72.  The RMSE of 2.86 means on average the predicted value will be off by 2.86 from the actual value.  R² of 0.15 which improved after adding more features but <1 means there may be more features that can affect the G3 grade.  The feature with the largest positive coefficient is internet which means people with internet access have higher G3 grade because they can access the study resources from the internet.  The feature with the largest negative coefficient is school support which surprised me because instead of having a positive effect on G3, it has the negative effect. 


# Neglected Feature: The Power of G1
#Add G1 to the x df
x = np.append(x, df_clean['G1'].values.reshape(-1, 1), axis=1)

#split
x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.2, random_state=42
)

#model fit
model = LinearRegression()
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)
print(f'New test r2 with G1: {r2_test}')

#Q: does a high R² here mean G1 is causing G3? Is this a useful model for identifying students who might struggle? What might educators need to do if they wanted to intervene early, before G1 is even available?

#A: It means G1 is the predictor of how student will do well on G3.  This may be useful in identifying students who might struggle.  The educators might want to address other features with negative coefficients before G1 is even available.
