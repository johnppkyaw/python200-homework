import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

# --- Pandas Review ---
# Pandas Q1
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(f"First 3 rows:\n {df[:3]}")
print(f"Shape:\n {df.shape}")
print(f"Data types:\n {df.dtypes}\n")

# Pandas Q2
print(f"Students who passed with grade above 80:\n {df[(df['passed']==True) & (df['grade'] > 80)]}\n")

# Pandas Q3
df['grade_curved'] = df['grade'] + 5
print(f"Updated df with grade_curved column:\n {df}\n")

# Pandas Q4
df['name_upper'] = df['name'].str.upper()
print(f"Name in uppercase:\n {df[['name', 'name_upper']]}\n")

# Pandas Q5
groupby = df.groupby("city").agg({"grade": "mean"})
print(f"Mean grade in each city:\n {groupby}\n")

# Pandas Q6
df = df.replace({"city": {"Austin": "Houston"}})
print(f"name and city columns:\n {df[['name', 'city']]}\n")

# Pandas Q7
df = df.sort_values(by="grade", ascending=False, ignore_index=True)
print(f"Sorted by grade in descending order:\n {df[:3]}\n")

# --- NumPy Review ---
# NumPy Q1
array_1d = np.array([10, 20, 30, 40, 50])
print(f"1D NumPy array:\n {array_1d}\n")

# NumPy Q2
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"2D array shape:\n {arr.shape}")
print(f"2D array size:\n {arr.size}\n")

# NumPy Q3
sliced_2d = arr[0:2, 0:2]
print(f"top-left 2x2 block:\n {sliced_2d}\n")

# NumPy Q4
zeros_arr = np.zeros((3, 4))
ones_arr = np.ones((2, 5))
print(f"3x4 array of zeros:\n {zeros_arr}")
print(f"2x5 array of ones:\n {ones_arr}\n")

# NumPy Q5
range_arr = np.arange(0, 50, 5)
print(f"range array:\n {range_arr}")
print(f"range array's shape:\n {range_arr.shape}")
print(f"range array's mean:\n {range_arr.mean()}")
print(f"range array's sum:\n {range_arr.sum()}")
print(f"range array's std:\n {range_arr.std()}\n")

# NumPy Q6
#200 random values drawn from a normal distribution with mean 0 and standard deviation 1
random_arr = np.random.normal(0, 1, size=(200))
print(f"random array's mean: \n {random_arr.mean()}")
print(f"random array's std: \n {random_arr.std()}")

# --- Matplotlib Review ---
# Matplotlib Q1
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
# plt.show()

# Matplotlib Q2
subjects = np.array(["Math", "Science", "English", "History"])
scores = np.array([88, 92, 75, 83])
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3
x1, y1 = np.array([[1, 2, 3, 4, 5], [2, 4, 5, 4, 5]])
x2, y2 = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
plt.scatter(x1, y1, color="blue", label="x1, y1")
plt.scatter(x2, y2, color="red", label="x2, y2")
plt.title("Scatter Plot of Two Datasets")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Matplotlib Q4
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, y)
ax1.set_title("Squares")
ax1.set(xlabel="x", ylabel="y")
ax2.bar(subjects, scores)
ax2.set_title("Subject Scores")
ax2.set(xlabel="Subjects", ylabel="Scores")
plt.tight_layout()
plt.show()

# --- Descriptive Statistics Review ---
# Descriptive Stats Q1
data = np.array([12, 15, 14, 10, 18, 22, 13, 16, 14, 15])
print(f"data's mean:\n {np.mean(data)}\n")
print(f"data's median:\n {np.median(data)}\n")
print(f"data's variance:\n {np.var(data)}\n")
print(f"data's std:\n {np.std(data)}\n")

# Descriptive Stats Q2
random_arr2 = np.random.normal(65, 10, 500)
plt.hist(random_arr2, 20)
plt.xlabel('Scores')
plt.ylabel('Count')
plt.title('Distribution of Scores')
plt.show()

# Descriptive Stats Q3
group_a = np.array([55, 60, 63, 70, 68, 62, 58, 65])
group_b = np.array([75, 80, 78, 90, 85, 79, 82, 88])
plt.title("Score Comparison")
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.show()

# Descriptive Stats Q4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.show()
#The mean provides more appropriate measure of central tendency for normal distribution and the median provides more appropriate measure of central tendency for the exponential distribution.

# Descriptive Stats Q8
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print(f"data1's mean: {np.mean(data1)}\n")
print(f"data1's median: {np.median(data1)}\n")
print(f"data1's mean: {np.mean(data2)}\n")
print(f"data1's median: {np.median(data2)}\n")

#Q: Why are the median and mean so different for data2?
#A: Because data2 is screwed to the right, compared to data1.


# --- Hypothesis Testing Review ---
# Hypothesis Q1
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}\n")

# Hypothesis Q2
if p_val < 0.05:
  print("The result is statistically significant.\n")
else:
  print("The result is not statistically significant.\n")

# Hypothesis Q3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
t_stat2, p_val2 = stats.ttest_rel(before, after)
print(f"t-statistic: {t_stat2:.3f}")
print(f"p-value: {p_val2:.6f}\n")

# Hypothesis Q4
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat3, p_val3 = stats.ttest_1samp(scores, 70)
print(f"t-statistic: {t_stat3:.3f}")
print(f"p-value: {p_val3:.6f}\n")

# Hypothesis Q5
t_stat4, p_val4 = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"t-statistic: {t_stat4:.3f}")
print(f"p-value: {p_val4:.6f}\n")

# Hypothesis Q6
print("The group_a's scores are less than the group_b's scores and this was not likely due to chance.")

# --- Correlation Review ---
# Correlation Review Q1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
print(f"Full correlation matrix: \n {corr_matrix}")
print(f"Correlation Coefficient at [0, 1]: {corr_matrix[0, 1]}\n")
#This is a strong positive relationship, as x increases, y increases.

# Correlation Review Q2
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
r, p = pearsonr(x, y)
print(f"correlation coefficient: {r}")
print(f"p-value: {p}\n")

# Correlation Review Q3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
result_corr_matrix = df.corr()
print(f"correlation matrix: \n {result_corr_matrix}\n")

# Correlation Review Q4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

# Correlation Review Q5
sns.heatmap(result_corr_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# --- Pipeline ---
# Pipeline Q1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
  series = pd.Series(arr, name="values")
  return series

def clean_data(series):
  return series.dropna()

def summarize_data(series):
  dict = {}
  dict["mean"] = series.mean()
  dict["median"] = series.median()
  dict["std"] = series.std()
  dict["mode"] = series.mode()[0]
  return dict

def data_pipeline(arr):
  series = create_series(arr)
  cleaned_series = clean_data(series)
  result_dict = summarize_data(cleaned_series)
  return result_dict

for key, value in data_pipeline(arr).items():
  print(f"{key}: {value:.3f}")

#This is more overhead because for processing the small data, it takes more time and resources than actual processing.

#Prefect becomes useful when data requires automatic update, a scheduled scripting or a visual dashboard.
