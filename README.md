SVM Binary Classification with Diagnosis Data

This project demonstrates how to build a Support Vector Machine (SVM) classifier for binary classification using a dataset with a ‘diagnosis’ column (e.g., Breast Cancer Dataset). It includes data preprocessing, model training, hyperparameter tuning, evaluation, and decision boundary visualization.

 Features

- Label encoding of categorical target (diagnosis)
- Feature scaling using StandardScaler
- Training SVM with both linear and RBF kernels
- Hyperparameter tuning using GridSearchCV
- Cross-validation performance evaluation
- 2D decision boundary visualization using PCA

 Dataset

The dataset should have a ‘diagnosis’ column with categorical values such as:
- ‘M’ (Malignant)
- ‘B’ (Benign)

Replace "your_dataset.csv" in the code with your actual dataset path.

 Requirements

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib

Install dependencies:
pip install pandas scikit-learn matplotlib

How to Run

1. Load and Preprocess Data
python
df = pd.read_csv("your_dataset.csv")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
```

2. Split, Scale, and Train SVM
python
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']
# Train-test split + standardize

3. Train SVM with Linear and RBF Kernels

4. Evaluate Model and Tune Hyperparameters
python
GridSearchCV with parameters:
C = [0.1, 1, 10, 100]
gamma = [1, 0.1, 0.01, 0.001]
kernel = 'rbf'

5. Visualize Decision Boundary
Using PCA to reduce to 2D for plotting.

 Output

- Accuracy scores
- Best parameters after Grid Search
- Cross-validation scores
- Visual plots of decision boundaries

 Notes

- PCA and 2D plotting are for visualization only.
- Final models should use full feature space for best performance.

Author

BOKKETI PRANITHA
