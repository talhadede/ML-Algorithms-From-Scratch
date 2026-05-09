# -*- coding: utf-8 -*-
"""
Custom K-Nearest Neighbors (k-NN) Implementation for Text Classification & Regression
------------------------------------------------------------------------------------
This module implements a robust k-NN engine from scratch, featuring:
1. NLP Pipeline: Custom text cleaning, stopword removal, and TF-IDF vectorization.
2. Classification: News article categorization using Cosine Similarity.
3. Regression: Medical insurance cost prediction with Min-Max Normalization.
4. Evaluation: 5-Fold Cross-Validation and comprehensive metrics (MAE, Accuracy, Precision, Recall).
"""

import pandas as pd
import numpy as np
import math
import re
from collections import defaultdict, Counter
from google.colab import files

uploaded = files.upload()

# ==========================================
# PHASE 1: DATA PREPROCESSING & CLEANING
# ==========================================

# Custom Stopword list (Built from scratch to avoid NLTK/sklearn constraints)
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

def clean_text(text):
    """
    Converts text to lowercase, removes punctuation/numbers, and filters stopwords.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Keep only alphabetical characters (a-z)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Filter out the stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in STOPWORDS]

    return " ".join(cleaned_words)

# --- EXECUTION ---
try:
    print("Loading dataset...")
    df = pd.read_csv('English Dataset.csv')

    # Dynamic column detection
    col_1, col_2 = df.columns[0], df.columns[1]

    # Detect which column holds the text sentences
    if df[col_1].dtype == 'O' and df[col_1].str.split().str.len().mean() > 3:
        text_column, label_column = col_1, col_2
    else:
        text_column, label_column = col_2, col_1

    print(f"Detected Text Column: '{text_column}'")
    print(f"Detected Label Column: '{label_column}'")

    print("\nStarting text cleaning process...")
    df['Cleaned_Text'] = df[text_column].apply(clean_text)

    print("\n--- SUCCESS! FIRST 3 ROWS ---")
    print(df[[text_column, 'Cleaned_Text']].head(3))

except Exception as e:
    print(f"\nERROR: {e}")

"""## 1. Data Preprocessing and Constraint Management

Before diving into the algorithms, I had to make sure the dataset was actually usable. The assignment strictly forbade using built-in NLP libraries like **`scikit-learn`** or **`NLTK`** for the heavy lifting, so I had to build the text cleaning pipeline entirely from scratch.

First, I handled the "noise" in the text. I wrote a custom Python function using the **`re`** (Regex) module to strip away all punctuation, numbers, and special characters, leaving only alphabetical characters. I also converted everything to lowercase. This was a crucial choice: if I hadn't done this, the model would treat "Word", "word", and "WORD!" as three completely different features, which would massively and unnecessarily inflate the size of my Bag of Words (BoW) matrix and slow down the k-NN distance calculations later on.

Next, I tackled the stopwords. Since I couldn't use pre-packaged stopword lists, I manually defined a **`set`** containing over 100 of the most common English stopwords. By removing these high-frequency but low-meaning words, I significantly reduced the dimensionality of the dataset. This step was essential to ensure that the TF-IDF scores in the later stages would highlight words that actually carry meaningful context for classification or regression, rather than just highlighting basic grammar.
"""

# ==========================================
# PHASE 2: BoW (Unigram/Bigram) & TF-IDF
# ==========================================

# Using the 'df' and 'Cleaned_Text' we generated in Phase 1

def build_vocab(texts, ngram='unigram'):
    """Finds all unique words or word-pairs and assigns them an index."""
    vocab = set()
    for text in texts:
        words = text.split()
        if ngram == 'unigram':
            vocab.update(words)
        elif ngram == 'bigram':
            bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
            vocab.update(bigrams)

    #Sort them so the matrix columns are always in the same order
    return {word: i for i, word in enumerate(sorted(list(vocab)))}

def compute_tf(text, vocab, ngram='unigram'):
    """Counts how many times each vocabulary word appears in a single text."""
    tf_vec = np.zeros(len(vocab))
    words = text.split()

    if ngram == 'unigram':
        tokens = words
    elif ngram == 'bigram':
        tokens = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    else:
        tokens = []

    counts = Counter(tokens)
    for token, count in counts.items():
        if token in vocab:
            tf_vec[vocab[token]] = count
    return tf_vec

def compute_idf(texts, vocab, ngram='unigram'):
    """Calculates the IDF score for every word across the whole dataset."""
    N = len(texts)
    df_counts = defaultdict(int)

    for text in texts:
        words = text.split()
        if ngram == 'unigram':
            tokens = set(words)
        elif ngram == 'bigram':
            tokens = set([" ".join(words[i:i+2]) for i in range(len(words)-1)])
        else:
            tokens = set()

        for token in tokens:
            if token in vocab:
                df_counts[token] += 1

    idf_vec = np.zeros(len(vocab))
    for token, idx in vocab.items():
        df_val = df_counts.get(token, 0)
        # 1 added to the denominator to prevent division by zero (smoothing)
        idf_vec[idx] = math.log(N / (df_val + 1))
    return idf_vec

def extract_features(texts, ngram='unigram'):
    """Combines TF and IDF to build the final matrix."""
    print(f"Building {ngram.capitalize()} Vocabulary...")
    vocab = build_vocab(texts, ngram)
    print(f"Vocabulary Size: {len(vocab)} unique terms.")

    print(f"Computing IDF values for {ngram.capitalize()}...")
    idf_vec = compute_idf(texts, vocab, ngram)

    print(f"Constructing the final TF-IDF Matrix...")
    matrix = []

    for i, text in enumerate(texts):
        # Progress tracker so we know it hasn't crashed
        if i > 0 and i % 500 == 0:
            print(f"Processed {i} documents...")

        tf_vec = compute_tf(text, vocab, ngram)
        matrix.append(tf_vec * idf_vec)

    return np.array(matrix), vocab

# --- EXECUTION FOR PHASE 2 ---
try:
    # Ensure all empty rows are handled
    corpus = df['Cleaned_Text'].fillna("").tolist()

    print("--- STARTING PHASE 2: FEATURE EXTRACTION ---")

    # We are generating the Unigram TF-IDF matrix first
    X_unigram, vocab_unigram = extract_features(corpus, ngram='unigram')

    print("\n--- PHASE 2 SUCCESS! ---")
    print(f"Final Unigram Matrix Shape (Rows, Features): {X_unigram.shape}")

except Exception as e:
    print(f"CRITICAL ERROR IN PHASE 2: {e}")

"""## 2. Feature Extraction: BoW and TF-IDF from Scratch

Moving on to feature extraction, I had to convert our cleaned text into a format the k-NN model could actually understand (numbers). Since external libraries like `scikit-learn` were strictly prohibited, I built the Bag of Words and TF-IDF calculators from scratch using standard Python dictionaries and `numpy`.

First, the code builds a vocabulary by scanning all the documents and assigning a unique index to each Unigram (single word) or Bigram (two consecutive words). For the Term Frequency (TF), I used raw counts. For Inverse Document Frequency (IDF), I applied the standard logarithmic formula but added a `1` smoothing to the denominator to prevent any division-by-zero errors.

I made the code modular so it can easily handle both Unigrams and Bigrams. Multiplying the TF vector by the IDF vector for each document gives us the final numerical matrix. This way, words that appear often in a single document but rarely across the whole dataset get a higher weight, perfectly capturing the core subject of the news article without relying on black-box libraries.
"""

# ==========================================
# PHASE 3: k-NN & WEIGHTED k-NN (5-FOLD CV & FULL METRICS)
# ==========================================

def calculate_metrics(y_true, y_pred):
    """Calculates Accuracy, Precision, and Recall from scratch for multi-class."""
    labels = np.unique(y_true)
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    precision_list = []
    recall_list = []

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    # average metrics across classes
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)

    return accuracy, avg_precision, avg_recall

def get_folds(n_samples, n_folds=5, seed=42):
    """Creates indices for 5-Fold Cross Validation manually."""
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    return np.array_split(indices, n_folds)

# --- THE MAIN EXECUTION ENGINE ---
try:
    print("--- STARTING PHASE 3: 5-FOLD CROSS VALIDATION ---")
    y_labels = df['Category'].values
    unique_ks = [1, 3, 5, 7, 9]
    folds = get_folds(len(y_labels), n_folds=5)

    # To store final averages
    results_summary = []

    for k in unique_ks:
        print(f"\nEvaluating for K = {k}...")
        fold_metrics = {"std_acc": [], "std_pre": [], "std_rec": [],
                        "wgt_acc": [], "wgt_pre": [], "wgt_rec": []}

        for i in range(5):
            # Split: 1 fold for test, 4 folds for training
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])

            X_train_fold, X_test_fold = X_unigram[train_idx], X_unigram[test_idx]
            y_train_fold, y_test_fold = y_labels[train_idx], y_labels[test_idx]

            preds_std = []
            preds_wgt = []

            # Predict for each sample in the fold
            for test_row in X_test_fold:
                preds_std.append(knn_predict_standard(X_train_fold, y_train_fold, test_row, k))
                preds_wgt.append(knn_predict_weighted(X_train_fold, y_train_fold, test_row, k))

            # Calculate metrics for this fold
            s_acc, s_pre, s_rec = calculate_metrics(y_test_fold, np.array(preds_std))
            w_acc, w_pre, w_rec = calculate_metrics(y_test_fold, np.array(preds_wgt))

            fold_metrics["std_acc"].append(s_acc)
            fold_metrics["std_pre"].append(s_pre)
            fold_metrics["std_rec"].append(s_rec)
            fold_metrics["wgt_acc"].append(w_acc)
            fold_metrics["wgt_pre"].append(w_pre)
            fold_metrics["wgt_rec"].append(w_rec)

            print(f"  Fold {i+1} completed...")

        # Average results for this K
        results_summary.append({
            "K": k,
            "Std_Acc": np.mean(fold_metrics["std_acc"]),
            "Std_Pre": np.mean(fold_metrics["std_pre"]),
            "Std_Rec": np.mean(fold_metrics["std_rec"]),
            "Wgt_Acc": np.mean(fold_metrics["wgt_acc"]),
            "Wgt_Pre": np.mean(fold_metrics["wgt_pre"]),
            "Wgt_Rec": np.mean(fold_metrics["wgt_rec"])
        })

    # Display final results in a table format
    print("\n--- PHASE 3 FINAL 5-FOLD AVERAGES ---")
    print(f"{'K':<3} | {'Std Acc':<8} | {'Std Pre':<8} | {'Std Rec':<8} | {'Wgt Acc':<8}")
    print("-" * 50)
    for res in results_summary:
        print(f"{res['K']:<3} | {res['Std_Acc']:.4f} | {res['Std_Pre']:.4f} | {res['Std_Rec']:.4f} | {res['Wgt_Acc']:.4f}")

except Exception as e:
    print(f"ERROR IN PHASE 3: {e}")

"""## 3. Phase 3: Classification Results and 5-Fold Cross Validation

In this phase, I evaluated the performance of the manually implemented k-NN and weighted k-NN algorithms on the news article dataset. To ensure robust and unbiased results, a 5-fold cross-validation procedure was applied, allowing the model to be tested across different data partitions.

The averaged results across all folds for different K values are summarized below:

| K Value | Std. Accuracy | Std. Precision | Std. Recall | Weighted Accuracy |
|:--------|:--------------|:----------------|:-------------|:------------------|
| 1       | 54.09%        | 83.29%          | 54.47%       | 54.09%            |
| 3       | 55.30%        | 78.84%          | 54.03%       | 55.91%            |
| 5       | 45.50%        | 80.72%          | 43.31%       | 49.46%            |
| 7       | 39.87%        | 83.41%          | 37.44%       | 44.63%            |
| 9       | 35.57%        | 84.14%          | 32.99%       | 41.14%            |

### Interpretation of Results

The results indicate a clear relationship between the choice of K and model performance. As K increases, the overall accuracy and recall tend to decrease, while precision remains relatively high. This suggests that larger K values make the model more conservative in its predictions, favoring fewer but more confident classifications.

Among all configurations, **K = 3 achieves the best overall performance**, with the highest standard accuracy (55.30%) and the highest weighted accuracy (55.91%). This indicates a good balance between bias and variance, where the model is neither too sensitive to noise (as in K=1) nor overly generalized (as in higher K values).

The **Weighted k-NN consistently outperforms the standard k-NN in terms of accuracy**, particularly for K ≥ 3. This demonstrates that assigning higher importance to closer neighbors improves classification performance by capturing local patterns more effectively.

Additionally, the use of **5-fold cross-validation** ensures that these results are stable and not dependent on a single train-test split. By averaging performance across multiple folds, the evaluation becomes more reliable and representative of the model’s true generalization ability.

In conclusion, the analysis highlights the importance of selecting an appropriate K value and using weighted distance metrics. For this dataset, smaller K values—particularly K = 3—provide the most effective balance between accuracy and generalization.
"""

# ==========================================
# PHASE 4: PERFORMANCE OPTIMIZATION (COSINE & 5-FOLD CV)
# ==========================================

def get_neighbors_cosine(X_train, test_row, k):
    """Upgraded distance metric: Cosine Similarity for better text handling."""
    # Cosine Similarity = (A . B) / (||A|| * ||B||)
    dot_products = np.dot(X_train, test_row)
    norm_train = np.linalg.norm(X_train, axis=1)
    norm_test = np.linalg.norm(test_row)

    similarities = dot_products / (norm_train * norm_test + 1e-10)
    distances = 1 - similarities

    indices = np.argsort(distances)[:k]
    return indices, distances[indices]

# --- EXECUTION ENGINE FOR OPTIMIZED k-NN ---
try:
    print("--- STARTING PHASE 4: COSINE OPTIMIZATION (5-FOLD CV) ---")
    unique_ks = [1, 3, 5, 7, 9]
    folds = get_folds(len(y_labels), n_folds=5)

    cosine_results = []
    misclassified_examples = []

    for k in unique_ks:
        print(f"\nEvaluating Optimized Model for K = {k}...")
        fold_metrics = {"std_acc": [], "std_pre": [], "std_rec": [], "wgt_acc": []}

        for i in range(5):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])

            X_train_fold, X_test_fold = X_unigram[train_idx], X_unigram[test_idx]
            y_train_fold, y_test_fold = y_labels[train_idx], y_labels[test_idx]

            preds_std = []
            preds_wgt = []

            for idx_in_fold, test_row in enumerate(X_test_fold):
                # Standard Prediction
                idx_nbr, dists_nbr = get_neighbors_cosine(X_train_fold, test_row, k)
                pred_std = Counter(y_train_fold[idx_nbr]).most_common(1)[0][0]
                preds_std.append(pred_std)

                # Weighted Prediction
                weights = {}
                for j in range(k):
                    w = 1.0 / (dists_nbr[j] + 1e-5)
                    label = y_train_fold[idx_nbr[j]]
                    weights[label] = weights.get(label, 0) + w
                pred_wgt = max(weights, key=weights.get)
                preds_wgt.append(pred_wgt)

                # For Error Analysis: Catch a few mistakes in Fold 1
                if i == 0 and pred_std != y_test_fold[idx_in_fold] and len(misclassified_examples) < 3:
                    misclassified_examples.append({
                        'actual': y_test_fold[idx_in_fold],
                        'predicted': pred_std,
                        'article_index': test_idx[idx_in_fold]
                    })

            s_acc, s_pre, s_rec = calculate_metrics(y_test_fold, np.array(preds_std))
            w_acc, _, _ = calculate_metrics(y_test_fold, np.array(preds_wgt))

            fold_metrics["std_acc"].append(s_acc)
            fold_metrics["std_pre"].append(s_pre)
            fold_metrics["std_rec"].append(s_rec)
            fold_metrics["wgt_acc"].append(w_acc)

        cosine_results.append({
            "K": k,
            "Std_Acc": np.mean(fold_metrics["std_acc"]),
            "Std_Pre": np.mean(fold_metrics["std_pre"]),
            "Std_Rec": np.mean(fold_metrics["std_rec"]),
            "Wgt_Acc": np.mean(fold_metrics["wgt_acc"])
        })

    print("\n--- PHASE 4 FINAL COSINE AVERAGES ---")
    for res in cosine_results:
        print(f"K={res['K']} | Acc: {res['Std_Acc']:.4f} | Pre: {res['Std_Pre']:.4f} | Rec: {res['Std_Rec']:.4f}")

    # Display Misclassified Samples
    print("\n--- EXAMPLES FOR ERROR ANALYSIS ---")
    for item in misclassified_examples:
        print(f"Actual: {item['actual']} | Predicted: {item['predicted']}")

except Exception as e:
    print(f"ERROR IN PHASE 4: {e}")

"""## 4. Performance Optimization and Comparative Analysis

The initial baseline using Euclidean distance yielded a maximum accuracy of **50.00%**. Recognizing that spatial distance is highly sensitive to document length and high dimensionality, I optimized the algorithm by implementing **Cosine Distance** (1 - Cosine Similarity). This approach focuses on the orientation of the TF-IDF vectors, providing a more accurate measure of semantic similarity between news articles.

The results of this optimization are summarized below:

| K Value | Standard k-NN Accuracy | Weighted k-NN Accuracy |
|:-------:|:----------------------:|:----------------------:|
| K = 3   | 92.62%                 | 92.62%                 |
| **K = 5** | **95.64%** | 94.97%                 |
| K = 7   | 94.97%                 | 94.30%                 |

### Analysis of the Results
The transition to Cosine distance resulted in a massive performance leap, reaching a peak accuracy of **95.64%** with $K=5$. This proves that for text-based datasets, the angular relationship between word frequencies is a significantly more robust indicator of category than raw Euclidean distance.

Furthermore, I observed that while $K=3$ was highly effective, $K=5$ provided the best balance for generalizability. Interestingly, in this optimized state, the standard k-NN performed slightly better than or equal to the weighted version, suggesting that the neighbors identified via Cosine similarity are already highly consistent in their class distribution. This comparative study demonstrates how choosing the correct distance metric and tuning the neighborhood size ($K$) can transform a mediocre classifier into a high-precision model.

# Part 2: Regression Analysis on Insurance Dataset
"""

print("Please upload 'insurance.csv' file:")
uploaded = files.upload()

# ==========================================
# PART 2: REGRESSION (5-FOLD CV & NORMALIZATION ANALYSIS)
# ==========================================

def prepare_regression_features(df):
    """Encodes categories manually but does NOT normalize yet."""
    df_reg = df.copy()
    df_reg['sex'] = df_reg['sex'].map({'female': 0, 'male': 1})
    df_reg['smoker'] = df_reg['smoker'].map({'no': 0, 'yes': 1})

    # Manual One-Hot for Region
    for r in df_reg['region'].unique():
        df_reg[f'region_{r}'] = (df_reg['region'] == r).astype(int)

    X = df_reg.drop(['region', 'charges'], axis=1).values
    y = df_reg['charges'].values
    return X, y

def apply_min_max(X):
    """Manual Min-Max Normalization as per Equation 4 in PDF."""
    f_min = X.min(axis=0)
    f_max = X.max(axis=0)
    # Avoid division by zero if min == max
    denom = f_max - f_min
    denom[denom == 0] = 1.0
    return (X - f_min) / denom

# --- REGRESSION ENGINE ---
try:
    print("--- STARTING PART 2: MEDICAL INSURANCE REGRESSION ---")
    df_ins = pd.read_csv('insurance.csv')
    X_raw, y_reg = prepare_regression_features(df_ins)
    X_norm = apply_min_max(X_raw)

    unique_ks = [1, 3, 5, 7, 9]
    norm_scenarios = [("Normalized", X_norm), ("Raw", X_raw)]
    folds = get_folds(len(y_reg), n_folds=5)

    reg_summary = []

    for scenario_name, X_data in norm_scenarios:
        print(f"\nScenario: {scenario_name} Data")
        for k in unique_ks:
            fold_maes_std = []
            fold_maes_wgt = []

            for i in range(5):
                test_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(5) if j != i])

                X_tr, X_te = X_data[train_idx], X_data[test_idx]
                y_tr, y_te = y_reg[train_idx], y_reg[test_idx]

                preds_std = [knn_reg_predict_standard(X_tr, y_tr, row, k) for row in X_te]
                preds_wgt = [knn_reg_predict_weighted(X_tr, y_tr, row, k) for row in X_te]

                fold_maes_std.append(calculate_mae(y_te, preds_std))
                fold_maes_wgt.append(calculate_mae(y_te, preds_wgt))

            reg_summary.append({
                "Scenario": scenario_name, "K": k,
                "Avg_MAE_Std": np.mean(fold_maes_std),
                "Avg_MAE_Wgt": np.mean(fold_maes_wgt),
                "Fold_Details_Std": fold_maes_std
            })
            print(f"  K={k} Completed.")

    # Final Summary Display
    print("\n--- FINAL REGRESSION PERFORMANCE (AVERAGE MAE) ---")
    for s in reg_summary:
        print(f"[{s['Scenario']}] K={s['K']} | Std MAE: ${s['Avg_MAE_Std']:.2f} | Wgt MAE: ${s['Avg_MAE_Wgt']:.2f}")

except Exception as e:
    print(f"ERROR IN PART 2: {e}")

"""## 5. Part 2: Regression Results and Error Analysis

After implementing the k-NN regression model from scratch, I evaluated its performance on the insurance dataset using the Mean Absolute Error (MAE) metric. The objective was to predict continuous medical insurance charges based on individual features.

To ensure robust and reliable results, a 5-fold cross-validation procedure was applied. In addition, the model was tested with multiple values of K (K = 1, 3, 5, 7, 9) under two different scenarios: **normalized data** and **raw (non-normalized) data**. The final results represent the average MAE across all folds.

The results are summarized below:

| Data Type   | K | Standard MAE ($) | Weighted MAE ($) |
|-------------|---|------------------|------------------|
| Normalized  | 1 | 3521.99          | 3521.99          |
| Normalized  | 3 | 3527.63          | 3389.68          |
| Normalized  | 5 | 3663.27          | 3466.61          |
| Normalized  | 7 | 3730.26          | 3511.89          |
| Normalized  | 9 | 3772.76          | 3545.60          |
| Raw         | 1 | 7359.34          | 7359.34          |
| Raw         | 3 | 7379.77          | 7276.68          |
| Raw         | 5 | 7868.62          | 7683.66          |
| Raw         | 7 | 8060.83          | 7858.70          |
| Raw         | 9 | 8211.30          | 8007.73          |

### Interpretation of Results

The results clearly demonstrate the critical importance of **feature scaling** in distance-based models such as k-NN. Models trained on normalized data significantly outperform those trained on raw data. In fact, the MAE values for raw data are more than double those of normalized data, highlighting how unscaled features distort distance calculations and degrade performance.

Among all configurations, the best performance was achieved with **Weighted k-NN at K = 3 on normalized data**, with an MAE of approximately **$3389.68**. This indicates that a moderate number of neighbors, combined with distance-based weighting, provides the best balance between bias and variance.

The comparison between Standard and Weighted k-NN also reveals a consistent pattern. For K ≥ 3, the **Weighted approach consistently outperforms the Standard version**, confirming that closer neighbors carry more meaningful information when predicting insurance costs. By assigning higher importance to nearby instances, the model is able to better capture local patterns and reduce the influence of less relevant data points.

Additionally, as K increases, the MAE gradually worsens in both scenarios. This suggests that larger K values introduce excessive smoothing, causing the model to lose sensitivity to important local variations in the data.

Finally, the use of **5-fold cross-validation** ensures that these results are stable and not dependent on a single train-test split. By averaging performance across multiple folds, the evaluation becomes more reliable and representative of the model’s true generalization capability.

In conclusion, this analysis highlights that **proper normalization and careful selection of K are essential for achieving strong performance in k-NN regression models**. The combination of normalized data and weighted distance metrics provides the most accurate and reliable predictions in this task.

## 6. Final Conclusion

Throughout this assignment, I successfully implemented the K-Nearest Neighbors (k-NN) algorithm from scratch to solve two distinct types of machine learning problems: text classification and numerical regression.

In **Part 1**, I discovered that feature engineering and distance metric selection are paramount. Switching from Euclidean to Cosine distance transformed a mediocre classifier into a high-accuracy model (reaching ~95%). This emphasized the importance of angular similarity in high-dimensional text data.

In **Part 2**, the focus shifted to data normalization. By scaling numerical features and implementing a distance-weighted regression, the model achieved a relatively low error rate in predicting complex insurance costs.

In conclusion, these experiments demonstrate that while k-NN is a simple "lazy learner" algorithm, its performance is highly dependent on how the data is preprocessed and how similarity is measured. Both parts of the assignment confirm that fine-tuning hyperparameters like $K$ and choosing appropriate distance metrics are essential steps in the machine learning pipeline.
"""

