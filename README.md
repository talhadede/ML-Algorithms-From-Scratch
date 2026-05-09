# ML-Algorithms-From-Scratch
Implementation of classical MachineLearning algorithms (KNN, ID3 Decision Trees) from scratch and comparative analysis of classification models.

# Machine Learning Algorithms: From Scratch to Application 🧠🚀

This repository contains a collection of Machine Learning projects focused on understanding the core mathematics behind classical algorithms. Instead of relying solely on high-level libraries, several algorithms (like **K-Nearest Neighbors** and **ID3 Decision Trees**) were built entirely **from scratch** to demonstrate a deep understanding of their inner workings.

## 📂 Project Structure & Core Modules

### 1. Dry Bean Classification (Model Comparison & Analysis)
A comprehensive comparative analysis of multiple classification algorithms to categorize dry beans based on their morphological features.
* **Source Files:** `dry_bean_model_comparison.py`, `dry_bean_classification_analysis.ipynb`
* **Algorithms Evaluated:** Support Vector Machines (SVM with RBF Kernel), Random Forest, Naive Bayes, and KNN.
* **Key Focus:** Preventing Data Leakage through rigorous pipeline design ("Scaling after Splitting") and utilizing 5-Fold Cross-Validation.
* **Result:** Achieved **~93.17% accuracy** using SVM, outperforming Naive Bayes by handling correlated geometric features.

### 2. ID3 Decision Tree (Implementation From Scratch)
Implementation of the **ID3 (Iterative Dichotomiser 3)** Decision Tree algorithm completely from scratch for image-based flower classification.
* **Source Files:** `id3_decision_tree_scratch.py`, `flower_classification_id3.ipynb`
* **Key Focus:** Entropy and Information Gain calculations mathematically implemented without `scikit-learn`.
* **Features:** * Custom recursive tree-building algorithm.
    * Extracted root-to-leaf decision rules translated into human-readable `IF-THEN` statements.
    * Custom confusion matrix and metric calculation (Accuracy, Precision, Recall, F1-Score).
* **Dataset Note:** Due to file size limitations, the raw image dataset is hosted externally.
    * **Download Dataset:** [Click here to download from Google Drive](https://drive.google.com/file/d/18m0MSatxOOPM7FkpbvVun20w7RWeNI_X/view)

### 3. Custom KNN Engine (Text Classification & Regression)
A robust, custom-built K-Nearest Neighbors engine capable of handling both complex Text Classification and Numerical Regression.
* **Source Files:** `custom_knn_engine.py`, `knn_text_and_regression_analysis.ipynb`
* **Part A (Text Classification):** Implemented **Cosine Similarity** from scratch to classify text data in high-dimensional spaces, boosting base accuracy to ~95%.
* **Part B (Regression):** Implemented a distance-weighted KNN regressor to predict continuous variables (insurance costs).
* **Key Focus:** Custom stop-word filtering, high-dimensional vector math, and hyperparameter ($K$) optimization.

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3
* **Core Libraries:** `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
* **ML Frameworks:** `scikit-learn` (used for preprocessing and baseline comparisons)
* **Concepts:** Distance Metrics (Cosine, Euclidean), Entropy, Information Gain, Data Leakage Prevention, Cross-Validation.

---
*Note: The Jupyter Notebooks (`.ipynb`) in this repository contain full execution logs, mathematical insights, and detailed data visualizations.*
