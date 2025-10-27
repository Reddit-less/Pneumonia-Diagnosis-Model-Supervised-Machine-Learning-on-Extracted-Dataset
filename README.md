# Pneumonia-Diagnosis-Model-Supervised-Machine-Learning-on-Extracted-Dataset
This project implements a supervised machine learning workflow to solve a binary classification problem: detecting pneumonia from a tabular dataset of extracted X-ray features. The model's objective is to learn the complex patterns correlating these input features to the final diagnosis.

How to run:

1 - Open the Notebook

2 - Launch the notebook in Google Colab. Datasets should already be stored in:
```
"/content/drive/MyDrive/AI/1CWK100"
```
**Core Technology, Algorithms & Libraries**

Core Libraries:

``` Pandas: Data ingestion, manipulation, and exploratory data analysis.

Scikit-learn (sklearn): The primary library for machine learning tasks.

NumPy: Fundamental numerical computation.

Matplotlib / Seaborn: Data visualization.

```

**Supervised Learning Algorithms:**
```

KNeighborsClassifier: A non-parametric, instance-based model that classifies data based on the majority class among its nearest neighbors

Decision Tree Classifier: A non-linear model for capturing complex interactions.

GaussianNB (Gaussian Naive Bayes): A probabilistic classifier based on Bayes' theorem.

SGDClassifier (Stochastic Gradient Descent): An efficient classifier for large-scale linear models.

```

**Machine Learning Workflow:**

The project follows a structured machine learning pipeline:

Data Preprocessing & Feature Engineering: The raw data is ingested and analyzed for class imbalance and feature distributions. The pipeline includes encoding the categorical target variable (Pneumonia: 'yes'/'no') into a binary format and applying feature scaling (e.g., StandardScaler) to normalize the numerical features; ensuring models are not biased by feature scale.

Model Training & Selection: The dataset is split into training and testing sets (train_test_split) to evaluate generalization. Multiple algorithms (Logistic Regression, Decision Tree, Random Forest, GaussianNB, SGDClassifier) are trained on the training data and then comparatively evaluated.
*
*Model Performance & Evaluation**

Model performance is not based on accuracy alone but on metrics critical to medical diagnosis:

Precision, Recall (Sensitivity), and F1-Score: These metrics are prioritized to balance the model's ability to avoid false positives (Precision) against its ability to find all true positive cases (Recall), which is crucial in a medical context.

Confusion Matrix: Used to visualize the model's performance and understand the specific types of errors (False Positives vs. False Negatives) being made by the final selected model.



**Literature Review & Context**

The project notebook concludes with a literature review and bibliography since it was done Artificial Intelligence module. This research provides academic context for the problem, referencing key papers on topics such as:

Deep learning applications in medical imaging (e.g., Radiologist-Level Pneumonia Detection on Chest X-Rays With Deep Learning).

Advanced model architectures (e.g., Vision Transformers, Capsule Networks).

Modern training methodologies (e.g., Self-Supervised Learning).

This review informed the project's scope and the selection of appropriate, industry-standard classification models.
