# --- Preprocessing ---
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import requests
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

#Preprocessing Question 1
#stratify=y argument ensures each species appears in similar proportions in both sets, making our evaluation fair.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"X_train's shape:{X.shape}")
print(f"X_test's shape:{X_test.shape}")
print(f"y_train's shape:{y_train.shape}")
print(f"y_test's shape:{y_test.shape}")

#Preprocessing Question 2
#split the data into training and test sets, and then fit the scaler only on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
#axis=0 for column-wise mean
print(X_test_scaled.mean(axis=0))

#Q: why you fit the scaler on X_train only
#A: To prevent data leakage, so it has not seen the test data yet.

# --- KNN ---

#KNN Question 1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

#KNN Question 2
knn_2 = KNeighborsClassifier(n_neighbors=5)
knn_2.fit(X_train_scaled, y_train)
preds_2 = knn_2.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, preds_2))
print(classification_report(y_test, preds_2))

#Q: does scaling improve performance, hurt it, or make no difference? Why might that be for this particular dataset?
#A: Scaling did not improve but rather hurt it. This is because all of its features are already in the same unit of measure that scaling can actually hurt the accuracy.

#KNN Question 3
def five_fold_cross_validation(k, X_train, y_train, cv):
  knn = KNeighborsClassifier(n_neighbors=k)
  cv_scores = cross_val_score(knn, X_train, y_train, cv=cv)
  print(cv_scores)           # accuracy on each fold
  print(f"Mean: {cv_scores.mean():.3f}")
  print(f"Std:  {cv_scores.std():.3f}")

five_fold_cross_validation(5, X_train, y_train, 5)

#Q: Is this result more or less trustworthy than a single train/test split, and why?
#A: This result is more trustworthy than single train/test split because it prevents the result from just being due to chance from 1 split as testing in the different groups multiple times gives the actual average.

#KNN Question 4
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
for each_k in k_values:
  print(f'k value: {each_k}')
  five_fold_cross_validation(each_k, X_train, y_train, 5)
#Q: which k you would choose and why
#A: Both k of 5 and 7 have both best average score but I would pick 7 for more stable model.

# --- Classifier Evaluation ---
#Classifier Evaluation Question 1
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
plt.savefig("./outputs/knn_confusion_matrix.png")
plt.close()

#Q: which pair of species does the model most often confuse (if any)?
#A: None

# --- The sklearn API: Decision Trees ---
# Decision Trees Question 1
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)
y_pred_3 = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_3)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred_3))

#Q: compare the Decision Tree accuracy to KNN. 
#A: Compared to Decision Tree (0.96),  KNN has higher accuracy (1.0).

#A: Given that Decision Trees don't rely on distance calculations, would scaled vs. unscaled data affect the result? 
#A: It's less sensitive to scale, the data would not affect the result.

# ---  Logistic Regression and Regularization ---
#Logistic Regression Question 1
#C is the inverse of regularization strength
#C=0.01
log_reg_full_0 = LogisticRegression(C=0.01, max_iter=1000, solver="newton-cg")
log_reg_full_0.fit(X_train_scaled, y_train)
print(f"C value: 0.01")
print(f"total size of all coefficients: {np.abs(log_reg_full_0.coef_).sum()}")
#C=1.0
log_reg_full_1 = LogisticRegression(C=1, max_iter=1000, solver="newton-cg")
log_reg_full_1.fit(X_train_scaled, y_train)
print(f"C value: 1")
print(f"total size of all coefficients: {np.abs(log_reg_full_1.coef_).sum()}")
#C=100
log_reg_full_100 = LogisticRegression(C=100, max_iter=1000, solver="newton-cg")
log_reg_full_100.fit(X_train_scaled, y_train)
print(f"C value: 100")
print(f"total size of all coefficients: {np.abs(log_reg_full_100.coef_).sum()}")
#Q: what happens to the total coefficient magnitude as C increases? What does this tell you about what regularization is doing?
#A: the total coefficient magnitude increases as C increases.  This means less regulation as C increases.

# --- PCA ---
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

#PCA Question 1
print(f'Shape of X_digits: {X_digits.shape}')
print(f'Shape of images: {images.shape}')
fig, axes = plt.subplots(1, 10, figsize=(16, 8))
for i in range(10):
  each_image = images[y_digits==i][0]
  axes[i].imshow(each_image, cmap='gray_r')
  axes[i].set_title(f"Label: {i}")
plt.savefig('./outputs/sample_digits.png')
plt.close()

#PCA Question 2
pca = PCA(svd_solver="randomized", random_state=0)
pca.fit(X_digits)
scores = pca.transform(X_digits)
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.savefig('./outputs/pca_2d_projection.png')
plt.close()
#Q: Do same-digit images tend to cluster together in this 2D space?
#A: Yes, the same-digit images tend to cluster together in this 2D space.

#PCA Question 3
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.title("Cumulative Explained Variance vs Number of Components")
plt.xlabel("Number of Components")
plt.ylabel("Total Variance Explained")
plt.savefig('./outputs/pca_variance_explained.png')
plt.close()
#Q: approximately how many components do you need to explain 80% of the variance?
#A: approximately 15 components needed to explain 80% of variance.

#PCA Question 4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_components_list = [0, 2, 5, 15, 40]


fig, axes = plt.subplots(5, 5, figsize=(14, 14))

for col_idx in range(5):
  each_image = images[y_digits==col_idx][0]
  axes[0, col_idx].imshow(each_image, cmap='gray_r')
  axes[0, col_idx].set_title(f"Label: {col_idx}, Original")
  #rows correspond to each n value and columns show those 5 digits
  for row_idx in range(1, len(n_components_list)):
    each_image = reconstruct_digit(col_idx, scores, pca, n_components_list[row_idx])
    axes[row_idx, col_idx].imshow(each_image, cmap='gray_r')
    axes[row_idx, col_idx].set_title(f"Label: {col_idx}, Components: {n_components_list[row_idx]}")
plt.tight_layout()
plt.savefig('./outputs/pca_reconstructions.png')
plt.close()

#Q: at what n do the digits become clearly recognizable, and does that match where the variance curve levels off?

#A: At n = 40.  Yes, it matches where the variance curve levels off.
