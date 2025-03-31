# SENTIMENT-ANALYSIS-WITH-NLP

**COMPANY**: CODETECH IT SOLUTIONS

**NAME**: TARIMELA SRINIVASA SOUMYA

**INTERN ID**:CT12WJVV

**DOMAIN**: MACHINE LEARNING

**BATCH DURATION**: JANUARY 5th,2025 to APRIL 5th,2025

**MENTOR NAME**: NEELA SANTHOSH

This Python program performs sentiment analysis on a dataset of text reviews using a Logistic Regression model. It involves several stages, including data loading, preprocessing, feature extraction, model training, evaluation, and visualization.

---

## Step 1: Importing Libraries

The program uses the following libraries:
- **Pandas** for data manipulation and analysis.
- **NumPy** for numerical operations.
- **Matplotlib and Seaborn** for data visualization.
- **Scikit-Learn (sklearn)** for machine learning tasks such as data splitting, feature extraction, model building, and evaluation.

---

## Step 2: Loading the Dataset

The dataset is loaded using the `pd.read_csv()` function from Pandas. It is assumed to be located at the specified path `D:\soumya\Acr.csv`. After loading, the program prints the dataset’s shape (number of rows and columns) and displays the first few rows using `df.head()`.

---

## Step 3: Handling Missing Values

The program checks for missing values using `df.isnull().sum()`. If the 'review' column contains any missing values, they are filled with empty strings using `fillna('')`. This ensures that the model doesn't encounter errors during text processing. Additionally, it converts all text data to strings using `astype(str)`.

A validation check ensures that the dataset is not empty after handling missing values. If it is empty, an exception is raised using `raise ValueError()`.

---

## Step 4: Validating Column Names

The code checks for the presence of the required columns `'reviews.text'` and `'reviews.rating'`, which are assumed to represent the text reviews and sentiment labels. If these columns are missing, an error is raised using `raise ValueError()`.

- **reviews.text**: Contains customer reviews in textual form.
- **reviews.rating**: Represents sentiment labels, typically 0 for negative and 1 for positive.

---

## Step 5: Data Splitting

The dataset is divided into training and testing sets using `train_test_split()` with an 80-20 split (`test_size=0.2`). The `random_state=42` ensures reproducibility. Additional checks ensure that neither the training nor testing set is empty, raising an exception if any set has no data.

---

## Step 6: Feature Extraction Using TF-IDF

Text data is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** using `TfidfVectorizer`. TF-IDF represents how important a word is to a document in a collection, reducing the impact of frequently used words like "the" and "is."  
- `stop_words='english'` removes common stop words.
- `max_features=5000` limits the vocabulary size to 5,000 words for computational efficiency.

The vectorizer is applied using `fit_transform()` for the training data and `transform()` for the test data.

---

## Step 7: Model Training

A **Logistic Regression** model is chosen for classification using `LogisticRegression()`. It uses:
- `max_iter=500` to allow more iterations for convergence.
- `solver='saga'`, which is efficient for large datasets.

The model is trained using the `fit()` method on the TF-IDF-transformed training data.

---

## Step 8: Making Predictions

The trained model predicts sentiment labels for the test data using `predict()`. The predictions are stored in `y_pred`.

---

## Step 9: Model Evaluation

The model's performance is evaluated using:
- **Accuracy Score**: Calculated using `accuracy_score()`, representing the ratio of correct predictions to total predictions.
- **Classification Report**: Displays precision, recall, and F1-score using `classification_report()`, providing detailed insights into model performance.
- **Confusion Matrix**: Constructed using `confusion_matrix()` and visualized using Seaborn’s `heatmap()`. It shows actual vs. predicted results, with counts of true positives, true negatives, false positives, and false negatives.

Labels are clearly represented using `'Negative'` and `'Positive'` for easy interpretation.

---

## Step 10: Visualization

The confusion matrix is visualized using Matplotlib and Seaborn. The plot uses a blue color map (`cmap='Blues'`) and displays the matrix using `annot=True` for clear annotations.

---

## Conclusion

This program is an end-to-end implementation of sentiment analysis using a Logistic Regression model. By preprocessing the data using TF-IDF, it effectively converts text into numerical features. Logistic Regression is a suitable choice for binary classification tasks like sentiment analysis. Additionally, clear error-handling mechanisms ensure robust performance across various datasets.

The use of accuracy score, classification report, and confusion matrix provides a comprehensive evaluation of model performance, while visualization enhances interpretability.
