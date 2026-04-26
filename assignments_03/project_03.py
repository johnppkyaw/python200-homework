import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)


#Task 1: Load and Explore
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(df.head())
print(df.describe())

features_of_interest = ['word_freq_free', 'char_freq_!', 'capital_run_length_total']

for i in range(3):
  feature = features_of_interest[i]
  ham_data = df[df['spam_label'] == 0][feature]
  spam_data = df[df['spam_label'] == 1][feature]
  plt.figure(figsize=(8, 6))
  plt.boxplot([ham_data, spam_data], tick_labels=['Ham', 'Spam'])
  plt.title(f'Distribution of {feature}')
  plt.ylabel(f'Percentage of {feature}')
  plt.savefig(f'./outputs/boxplot_of_{feature}.png')
  plt.close()

print(df['word_freq_free'].describe)

#It's heavily skewed to zero because most values are zero.  The features are measured in completely different units, which causes different ranges.  Because of these the models will likely be biased on the data unless we scaled the data first.

#Task 2: Prepare Your Data
#Separate the features to X and spam_label to Y
#Split them into trained and test data
#Use the standardscaler to scale both train and test data.
X = df.drop('spam_label', axis=1)
y = df['spam_label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
# learns mean and std from training data only
X_train_scaled = scaler.fit_transform(X_train)
# applies the same scaling to test data
X_test_scaled  = scaler.transform(X_test)

#PCA preprocessing and fitting only train scaled data
pca = PCA()
pca.fit(X_train_scaled)

#cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print(cumulative_variance)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.title("Cumulative Explained Variance vs Number of Components")
plt.xlabel("Number of Components")
plt.ylabel("Total Variance Explained")
plt.savefig('./outputs/email_pca_variance_explained.png')
plt.close()

n = 0
for i, variance in enumerate(cumulative_variance):
  if variance >= 0.90:
    n = i + 1
    break

print(f"Number of components to reach 90% variance (n): {n}")

#PCA-reduced arrays
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]


#Task 3: A Classifier Comparison

#KNeighborsClassifier on unscaled
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
print("KNN on unscaled")
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

#KNeighborsClassifier on scaled and PCA-reduced
knn_2s = KNeighborsClassifier(n_neighbors=5)
knn_2s.fit(X_train_scaled, y_train)
preds_2s = knn_2s.predict(X_test_scaled)
print("KNN on scaled")
print("Accuracy:", accuracy_score(y_test, preds_2s))
print(classification_report(y_test, preds_2s))

#KNeighborsClassifier on scaled and PCA-reduced
knn_2p = KNeighborsClassifier(n_neighbors=5)
knn_2p.fit(X_train_pca, y_train)
preds_2p = knn_2p.predict(X_test_pca)
print("KNN on PCA-reduced")
print("Accuracy:", accuracy_score(y_test, preds_2p))
print(classification_report(y_test, preds_2p))

#DecisionTreeClassifier(random_state=42) -- before settling on a final depth, try max_depth values of 3, 5, 10, and None (unlimited)
depths = [3, 5, 10, None]
for d in depths:
  dt = DecisionTreeClassifier(max_depth=d, random_state=42)
  dt.fit(X_train_scaled, y_train)
  train_pred = dt.predict(X_train_scaled)
  test_pred = dt.predict(X_test_scaled)
  train_acc = accuracy_score(y_train, train_pred)
  test_acc = accuracy_score(y_test, test_pred)
  print(f'Max Depth: {d}')
  print(f'Training accuracy: {train_acc:.3f}')
  print(f'Test accuracy: {test_acc:.3f}')

#As the depth increases, the accuracies increases but so does the gap between training and test accuracy leading to overfitting.
#I would use the depth of 5, as it improves the accuracies and also has less gap than depth 10.

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)
test_pred = dt.predict(X_test_scaled)
print("DecisionTreeClassifier with depth of 5")
print("Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

#RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
test_pred_rf = rf.predict(X_test)
print("RandomForestClassifier")
print("Accuracy:", accuracy_score(y_test, test_pred_rf))
print(classification_report(y_test, test_pred_rf))


dt_feature_importances = pd.Series(dt.feature_importances_, index=X_train.columns)
dt_feature_importances = dt_feature_importances.sort_values(ascending=False)
dt_top_features = dt_feature_importances.head(10)
print("DT's top 10 features: ")
print(dt_top_features)

rf_feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
rf_feature_importances = rf_feature_importances.sort_values(ascending=False)
rf_top_features = rf_feature_importances.head(10)
print("RF's top 10 features: ")
print(rf_top_features)

plt.figure(figsize=(10, 8))
rf_top_features.plot(kind='bar', color='blue')
plt.title("RF's Top 10 Features for Spam Detection")
plt.xlabel('Email Features (Words/Characters)')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('./outputs/feature_importances.png')
plt.close()

#They agree on most features but do have a few features that were not the same.  Both models likely flagged the $ and ! characters, so the results match my intuition.

#LogisticRegression(C=1.0, max_iter=1000, solver='liblinear') trained on the scaled data, and again on the PCA-reduced data -- compare the two
#Trained on scaled data
log_reg_full_1 = LogisticRegression(C=1, max_iter=1000, solver="liblinear")
log_reg_full_1.fit(X_train_scaled, y_train)
pred_log_reg_1 = log_reg_full_1.predict(X_test_scaled)
print("LogisticRegression on scaled")
print("Accuracy:", accuracy_score(y_test, pred_log_reg_1))
print(classification_report(y_test, pred_log_reg_1))

#Trained on PCA-reduced data
log_reg_full_2 = LogisticRegression(C=1, max_iter=1000, solver="liblinear")
log_reg_full_2.fit(X_train_pca, y_train)
pred_log_reg_2 = log_reg_full_2.predict(X_test_pca)
print("LogisticRegression on PCA-reduced data")
print("Accuracy:", accuracy_score(y_test, pred_log_reg_2))
print(classification_report(y_test, pred_log_reg_2))

#Q: Which model performs best? For the classifiers where you compared PCA vs. non-PCA, which worked better -- and does that match your hypothesis from Task 2? For a spam filter specifically, is accuracy the right metric to optimize -- or would you rather minimize false positives (legitimate email marked as spam) or false negatives (spam that gets through)?

#A:Random Forest Classifier is the best model with accuracy of 94.6%. When I compared PCA vs non-PCA, it did not match with the hypothesis from Task 2.  The scaled non-PCA models worked better than PCA models.  For a spam filter, I would rather minimize false positives (legitimate email marked as spam) than accuracy or false negatives because we don't want to have the risk of the important emails (eg, payment emails or client emails) being marked as spam.

#The confusion matrix of Random Forest Classifier
cm = confusion_matrix(y_test, test_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot()
plt.title('Confusion Matrix of Random Forest Classifier')
plt.savefig('./outputs/best_model_confusion_matrix.png')
plt.close()

#Q: which type of error does your best model make more often?
#A: It's letting the spam email thru more often than blocking the legitimate email, more false negatives than false positives.

#Task 4: Cross-Validation
models = [
    {
        'model': knn, 
        'name': "KNN unscaled",
        'X': X_train
    },
    {
        'model': knn_2s,
        'name': "KNN scaled",
        'X': X_train_scaled
    },
    {
        'model': knn_2p, 
        'name': "KNN PCA-reduced",
        'X': X_train_pca
    },
    {
        'model': dt, 
        'name': "Decision Tree Classifier (depth 5)",
        'X': X_train_scaled
    },
    {
        'model': rf, 
        'name': "Random Forest Classifier",
        'X': X_train
    },
    {
        'model': log_reg_full_1, 
        'name': "Logistic Regression scaled",
        'X': X_train_scaled
    },
    {
        'model': log_reg_full_2, 
        'name': "Logistic Regression PCA-reduced",
        'X': X_train_pca
    }
]

#For each, print the mean and standard deviation of the fold scores. 
for each_model in models:
  cv_scores = cross_val_score(each_model['model'], each_model['X'], y_train, cv=5)
  print(f"{each_model['name']} Mean: {cv_scores.mean():.3f}")
  print(f"{each_model['name']} Std: {cv_scores.std():.3f}")
  
#Q: Which model is the most accurate? Which is the most stable (lowest variance across folds)? Does the ranking match what you saw with the single train/test split?
#A: Random forest classifier is the most accurate model and the logistic regression with PCA-reduced data is the most stable model.  The ranking mostly match with what I saw with the single train/test split.

#Task 5: Building a Prediction Pipeline

#Random Forest Classifier pipeline
rf_pipeline = Pipeline([
  ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf_pipeline = rf_pipeline.predict(X_test)
print("Random Forest Classifier pipeline")
print(classification_report(y_test, y_pred_rf_pipeline))


#Logistic Regression scaled pipeline
log_reg_pipeline = Pipeline([
  ("scaler", StandardScaler()),
  ("classifier", LogisticRegression(C=1, max_iter=1000, solver="liblinear"))
])
log_reg_pipeline.fit(X_train, y_train)
y_pred_log_reg_pipeline = log_reg_pipeline.predict(X_test)
print("Logistic Regression scaled pipeline")
print(classification_report(y_test, y_pred_log_reg_pipeline))

#Q:Comment on your pipelines: do they have the same structure? Why or why not? What is the practical value of packaging a model this way, especially when handing it off to someone else or deploying it?
#A: Both pipelines have the same structure but different number of components because Logistic Regression pipeline includes StandardScalar step while Random Forest pipeline does not. The advantage of packaging a model this way is so, that when we work as a team, it prevents the data leakage.
