# %%
#!pip install pandas numpy opencv-python matplotlib
#!pip install scikit-learn seaborn

# %%
import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import copy

# %%
def calculate_metrics(y_true, y_pred, classes):

    extended_classes = classes + ["Unknown"]
    cm = confusion_matrix(y_true, y_pred, labels=extended_classes)
    
    num_classes = len(classes)
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    total_samples = len(y_true) 
    
    overall_accuracy = sum(cm[i, i] for i in range(num_classes)) / total_samples
    
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print("-" * 80)
    print(f"{'Class':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 80)
    
    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = total_samples - TP - FP - FN
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) if total_samples > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        print(f"{classes[i]:<15} | {accuracy:.4f}     | {precision:.4f}     | {recall:.4f}    | {f1:.4f}")
        
    print("-" * 80)
    print(f"{'Macro Average':<15} | {np.mean(accuracies):.4f}     | {np.mean(precisions):.4f}     | {np.mean(recalls):.4f}    | {np.mean(f1_scores):.4f}")
    print("-" * 80)

# %%
# Configuration and Paths
DATA_FOLDER = 'flowers'
TRAIN_PATH = os.path.join(DATA_FOLDER, 'train')
TEST_PATH = os.path.join(DATA_FOLDER, 'test')
VAL_PATH = os.path.join(DATA_FOLDER, 'validation')

# Get the list of flower species
flower_species = sorted(os.listdir(TRAIN_PATH))
print(f"Total Species: {len(flower_species)}")
print("Species List:", flower_species)

# %%
def extract_features(image_path):
    # 1.Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 2.Preprocessing: Resize to 100x100 for consistency
    image = cv2.resize(image, (100, 100))
    
    # 3.Color Features: Convert to HSV and get mean values
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv_image[:, :, 0])
    mean_saturation = np.mean(hsv_image[:, :, 1])
    mean_value = np.mean(hsv_image[:, :, 2])
    
    # 4.Shape Features: Edge Density
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    edge_density = np.sum(edges > 0) / (100 * 100)
    
    # 5.Texture Features: Standard Deviation
    texture_score = np.std(gray_image)
    
    return {
        'mean_hue': mean_hue,
        'mean_saturation': mean_saturation,
        'mean_value': mean_value,
        'edge_density': edge_density,
        'texture_score': texture_score
    }

# %%
def create_feature_dataset(source_path):
    """
    Iterates through folders, extracts features from each image, 
    and returns a formatted Pandas DataFrame.
    """
    data_records = []
    # ort list of flower categories
    categories = sorted(os.listdir(source_path))
    
    print(f"--- Processing directory: {source_path} ---")
    
    for category in categories:
        category_path = os.path.join(source_path, category)
        # Skip if not a directory
        if not os.path.isdir(category_path):
            continue
            
        print(f"Current Category: {category}...")
        image_list = os.listdir(category_path)
        
        for img_name in image_list:
            img_full_path = os.path.join(category_path, img_name)
            
            # Use our previously defined extraction function
            features = extract_features(img_full_path)
            
            if features is not None:
                # Add the 'label' as target variable for classification
                features['label'] = category
                data_records.append(features)
                
    # Create the final feature matrix
    return pd.DataFrame(data_records)

# --- EXECUTION ---
train_df = create_feature_dataset(TRAIN_PATH)

print("\nProcess Completed Successfully!")
print(f"Total instances in training set: {len(train_df)}")

# %%
print("--- First 5 Rows of the Dataset ---")
print(train_df.head())

# Check the distribution
print("\n--- Class Distribution (Instance count per species) ---")
print(train_df['label'].value_counts())

# Basic statistical summary
print("\n--- Statistical Summary ---")
print(train_df.describe())

# %%
# We will use 'qcut' to divide features into 3 equal sized categories (bins)
# 0: Low, 1: Medium, 2: High

def discretize_data(df):
    """
    Converts continuous numerical features into categorical bins 
    using equal-frequency binning.
    """
    # keep the original numeric data
    discretized_df = df.copy()
    
    # Identify numerical columns (exclude the 'label' column)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        # Divide each feature into 3 bins: Low, Mid, High
        # labels=False returns 0, 1, 2 which is easier for ID3 logic
        discretized_df[col] = pd.qcut(df[col], q=3, labels=["Low", "Mid", "High"], duplicates='drop')
    
    return discretized_df

# --- EXECUTION ---
# Apply discretization to our training set
train_df_discretized = discretize_data(train_df)

print("--- Discretized Data (First 5 Rows) ---")
print(train_df_discretized.head())

# %%
def calculate_entropy(data_label_col):
    """
    Calculates the entropy of a given list of labels.
    Entropy measures the 'disorder' or 'impurity' in the data.
    """
    # Count the number of instances for classe
    label_counts = data_label_col.value_counts()
    total_instances = len(data_label_col)
    
    entropy = 0
    for count in label_counts:
    
        probability = count / total_instances
        # Entropy formula: -p * log2(p)
        entropy -= probability * math.log2(probability)
        
    return entropy

# --- TEST ---
# calculate the starting entropy of our 6000 images
initial_entropy = calculate_entropy(train_df_discretized['label'])
print(f"Initial Entropy of the Dataset: {initial_entropy:.4f}")

# %%
def calculate_information_gain(df, feature_name, label_name='label'):
    """
    Calculates the Information Gain of a specific feature.
    It tells us how much entropy is reduced by splitting on this feature.
    """
    # 1. Calculate the total entropy before the split (Parent Entropy)
    total_entropy = calculate_entropy(df[label_name])
    
    # 2. Get the unique categories of the feature (Low, Mid, High)
    feature_values = df[feature_name].unique()
    
    # 3. Calculate the weighted average entropy after the split (Child Entropy)
    weighted_entropy = 0
    total_instances = len(df)
    
    for value in feature_values:
        # Create a subset for each category
        subset = df[df[feature_name] == value]
        subset_entropy = calculate_entropy(subset[label_name])
        
        # Weighted entropy: proportion of items *entropy of that subset
        weight = len(subset) / total_instances
        weighted_entropy += weight * subset_entropy
        
    # Information Gain = Entropy before split - Entropy after split
    info_gain = total_entropy - weighted_entropy
    return info_gain

# --- TEST & ANALYSIS ---
features_to_test = ['mean_hue', 'mean_saturation', 'mean_value', 'edge_density', 'texture_score']

print("--- Information Gain Analysis ---")
for feature in features_to_test:
    gain = calculate_information_gain(train_df_discretized, feature)
    print(f"Gain for {feature}: {gain:.4f}")

# %%
def build_id3_tree(df, original_df, features, target_attribute_name="label", parent_node_class=None):
    """
    Recursively builds the ID3 Decision Tree.
    """
    # 1. Stopping Criterion: If all target values are the same, return this value
    unique_classes = np.unique(df[target_attribute_name])
    if len(unique_classes) <= 1:
        return unique_classes[0]

    # 2. Stopping Criterion: If dataset is empty, return the majority class of the parent
    elif len(df) == 0:
        return np.unique(original_df[target_attribute_name])[
            np.argmax(np.unique(original_df[target_attribute_name], return_counts=True)[1])
        ]

    # 3. Stopping Criterion: If no features left, return the majority class
    elif len(features) == 0:
        return parent_node_class

    # 4. Build the tree
    else:
        # Set the default majority class for this node
        parent_node_class = np.unique(df[target_attribute_name])[
            np.argmax(np.unique(df[target_attribute_name], return_counts=True)[1])
        ]

        # Select the feature that best splits the dataset (Highest Information Gain)
        item_values = [calculate_information_gain(df, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create the tree structure
        tree = {best_feature: {}}

        # Remove the best feature from the feature list
        remaining_features = [i for i in features if i != best_feature]

        # Grow a branch for each value of the best feature (Low, Mid, High)
        for value in np.unique(df[best_feature]):
            # Split the dataset
            sub_data = df[df[best_feature] == value]
            
            # Call the algorithm recursively
            subtree = build_id3_tree(sub_data, df, remaining_features, target_attribute_name, parent_node_class)
            
            # Add the subtree to the main tree
            tree[best_feature][value] = subtree

        return tree

# --- EXECUTION ---
# List of features to use for the tree
features_list = ['mean_hue', 'mean_saturation', 'mean_value', 'edge_density', 'texture_score']

print("Building the ID3 Tree... This may take a few seconds.")
flower_decision_tree = build_id3_tree(train_df_discretized, train_df_discretized, features_list)

print("\nTree construction successful!")

# %%
def predict(tree, instance):
    """
    Traverses the ID3 dictionary tree to predict the class of a given instance.
    """
    # 1. Base case: If the current node is a leaf , return it
    if not isinstance(tree, dict):
        return tree

    # 2. Get the feature name from the current node
    root_feature = list(tree.keys())[0]
    
    # 3. Get the value of this feature from the test image (like 'Low')
    feature_value = instance[root_feature]

    # 4. Check if a branch exists for this specific value
    if feature_value in tree[root_feature]:
        # Recursive call to move deeper into the subtree
        return predict(tree[root_feature][feature_value], instance)
    else:
        # If the value wasn't seen in training, return 'Unknown'
        return "Unknown"

# %%
# --- 1: Process Validation Images ---
print("Extracting features from validation set... This may take a moment.")

validation_df_raw = create_feature_dataset(VAL_PATH)

# --- 2: Apply Discretization ---
validation_df_discretized = discretize_data(validation_df_raw)

# --- 3: Calculate Performance ---
def calculate_accuracy(tree, df):
    """
    Calculate accuracy, returning 0.0 if the dataset is empty.
    """
    if df is None or len(df) == 0:
        return 0.0
    
    correct_hits = 0
    for _, row in df.iterrows():
        if predict(tree, row) == row['label']:
            correct_hits += 1
            
    return (correct_hits / len(df)) * 100

# --- EXECUTION ---
val_acc = calculate_accuracy(flower_decision_tree, validation_df_discretized)

print("\n" + "="*40)
print(f"VAL ACCURACY: %{val_acc:.2f}")
print("="*40)

# %% [markdown]
# ### Phase 1: Baseline Model Construction and Evaluation
# 
# **Methodology:**
# For the baseline model, we used basic color features (Mean Hue, Saturation, and Value) along with Edge Density for texture. To make these continuous features work with the ID3 algorithm, we discretized them into 3 simple bins (Low, Mid, High).
# 
# **Performance Reporting & Analysis:**
# We evaluated this first tree on the validation set to see how it performs. The results below show a clear case of **Underfitting**. It turns out that splitting the data into just 3 bins and using basic features doesn't capture enough detail to tell 10 different flower species apart. Because the tree couldn't learn the complex patterns from such limited options, the precision and recall scores ended up being quite low across the board.

# %%
def plot_confusion_matrix(tree, df, classes):
    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        # Get actual and predicted labels
        actual = row['label']
        prediction = predict(tree, row)
        
        y_true.append(actual)
        y_pred.append(prediction)
    
    # Generate the matrix
    # Note: 'Unknown' predictions are handled by including them in labels if necessary
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Initial ID3 Model (Accuracy: 29.60%)', fontsize=15)
    plt.ylabel('Actual Species', fontsize=12)
    plt.xlabel('Predicted Species', fontsize=12)
    plt.xticks(rotation=45)
    
    # Save for report
    plt.savefig('initial_confusion_matrix.png', bbox_inches='tight')
    print("Confusion Matrix has been saved as 'initial_confusion_matrix.png'")

plot_confusion_matrix(flower_decision_tree, validation_df_discretized, flower_species)

# %%
y_true_p1 = []
y_pred_p1 = []
for _, row in validation_df_discretized.iterrows(): 
    y_true_p1.append(row['label'])
    y_pred_p1.append(predict(flower_decision_tree, row))

print("\n--- PHASE 1: Validation Performance Metrics ---")
calculate_metrics(y_true_p1, y_pred_p1, flower_species)

# %% [markdown]
# ### 1. Executive Summary
# We tested the initial ID3 Decision Tree on 10 flower species using a training set of 6,000 images. The baseline model got a validation accuracy of 29.60%. This is better than a random guess, but it clearly shows that our current features are not enough for a good classification.
# 
# ### 2. Methodology (Initial Configuration)
# * **Classifier:** ID3 Algorithm.
# * **Preprocessing:** Resized all images to $100 \times 100$ pixels.
# * **Features Used:** Mean Hue, Mean Saturation, Mean Value, Edge Density, and Texture Score.
# * **Discretization:** Divided the continuous features into 3 equal bins (Low, Mid, High).
# 
# ### 3. Performance Analysis (Confusion Matrix)
# Looking at the confusion matrix, we noticed a few main points:
# * **Best Performing Classes:** Iris, Daisy, and Sunflower had the highest recognition rates. This is probably because of their distinct colors.
# * **Critical Failures:** Tulip accuracy was terrible, hitting only 1%. The model usually confused it with sunflowers and irises. This means color alone is not enough to classify tulips.
# * **Species Ambiguity:** Dandelion and Daisy were heavily confused. Since they share similar yellow colors and edge densities, the tree couldn't separate them well.
# 
# ### 4. Root Cause of Low Accuracy
# * **Low Granularity in Discretization:** Using only 3 bins causes too much information loss. The tree doesn't have enough detail to make good splits.
# * **Lack of Geometric Features:** The model only uses color and basic texture right now. It completely ignores the actual shape of the flowers.
# 
# ### 5. Proposed Improvements (Phase 2)
# To fix these issues and improve accuracy in the next phase, we will do the following:
# * **Feature Expansion:** Add Aspect Ratio to capture geometric differences between species.
# * **Increased Binning:** Change the discretization from 3 to 5 bins to keep more data details.
# * **Model Optimization:** We will need to implement a pruning technique later to prevent overfitting, since we are adding more features and bins to the tree.

# %%
def extract_features_v2(image_path):
    """
    Enhanced feature extraction including geometric aspect ratio.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Calculate Geometric Feature: Aspect Ratio (Width / Height)
    height, width, _ = img.shape
    aspect_ratio = width / height
    
    # Preprocessing
    img_resized = cv2.resize(img, (100, 100))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Color Features (HSV Mean Values)
    mean_hue = np.mean(hsv[:, :, 0])
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_value = np.mean(hsv[:, :, 2])
    
    # Texture/Structural Feature: Edge Density (Canny)
    edges = cv2.Canny(img_resized, 100, 200)
    edge_density = np.sum(edges > 0) / (100 * 100)
    
    return {
        'mean_hue': mean_hue,
        'mean_saturation': mean_saturation,
        'mean_value': mean_value,
        'edge_density': edge_density,
        'aspect_ratio': aspect_ratio
    }

# %%
def create_dataset_v2(source_path):
    """
    Constructs the feature matrix using the V2 extraction method.
    """
    data_list = []
    categories = sorted(os.listdir(source_path))
    
    print(f"Starting Phase 2 processing for: {source_path}")
    
    for category in categories:
        category_path = os.path.join(source_path, category)
        if not os.path.isdir(category_path):
            continue
            
        print(f"Processing Category: {category}...")
        images = os.listdir(category_path)
        
        for img_name in images:
            full_path = os.path.join(category_path, img_name)
            features = extract_features_v2(full_path)
            
            if features is not None:
                features['label'] = category
                data_list.append(features)
                
    return pd.DataFrame(data_list)

print("--- Building Enhanced Training Dataset ---")
train_df_v2 = create_dataset_v2(TRAIN_PATH)
print(f"Phase 2 Construction Complete. Total Instances: {len(train_df_v2)}")

# %%
def discretize_v2(df, bin_count=7):
    """
    Improved discretization that handles features with many duplicate values.
    """
    discretized_df = df.copy()
    numeric_features = df.select_dtypes(include=[np.number]).columns
    
    print("Starting robust discretization...")
    
    for col in numeric_features:
        try:
            # Try to create 'bin_count' number of bins
            # labels=False returns integers (0, 1, 2...) instead of interval objects
            bins = pd.qcut(df[col], q=bin_count, labels=False, duplicates='drop')
            
            # Map integers to "Level_X" strings
            discretized_df[col] = bins.map(lambda x: f"Level_{int(x)}")
            
        except Exception as e:
            print(f"Warning: Could not discretize column '{col}' into {bin_count} bins. Using simpler split.")
            # Fallback for columns with very few unique values
            discretized_df[col] = pd.cut(df[col], bins=3, labels=["Level_0", "Level_1", "Level_2"])
            
    return discretized_df

# Applying the fixed discretization
print("Applying improved 7-bin discretization for Phase 2...")
train_df_discretized_v2 = discretize_v2(train_df_v2, bin_count=7)

# Preview the results to ensure 'Level_X' format is correct
print("\n--- Discretized Data Preview (Phase 2) ---")
print(train_df_discretized_v2.head())

# %%
# Updated features list (including aspect_ratio)
features_list_v2 = ['mean_hue', 'mean_saturation', 'mean_value', 'edge_density', 'aspect_ratio']

print("Building the Enhanced ID3 Tree (Phase 2)...")
# This might take slightly longer because of the 7-bin depth
flower_tree_v2 = build_id3_tree(train_df_discretized_v2, train_df_discretized_v2, features_list_v2)

print("Phase 2 Tree Construction Successful!")

# %%
# --- STEP A: Extract V2 features from Validation Set ---
print("Extracting Phase 2 features from validation set...")
val_df_raw_v2 = create_dataset_v2(VAL_PATH)

# --- STEP B: Apply 7-bin Discretization to Validation Set ---
print("Discretizing validation set (Phase 2)...")
val_df_discretized_v2 = discretize_v2(val_df_raw_v2, bin_count=7)

# --- STEP C: Calculate New Accuracy ---
print("Evaluating Phase 2 Performance...")
accuracy_v2 = calculate_accuracy(flower_tree_v2, val_df_discretized_v2)

print("\n" + "="*40)
print(f"PHASE 2 VALIDATION ACCURACY: %{accuracy_v2:.2f}")
print("="*40)

# %%
# Training set üzerindeki başarıyı ölçelim
train_acc_v2 = calculate_accuracy(flower_tree_v2, train_df_discretized_v2)
print(f"Phase 2 Training Accuracy: %{train_acc_v2:.2f}")
print(f"Phase 2 Validation Accuracy: %{accuracy_v2:.2f}")

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Phase 2 tahminlerini toplama
y_true_p2 = []
y_pred_p2 = []

print("Analyzing Phase 2... Please wait.")
for _, row in val_df_discretized_v2.iterrows():
    y_true_p2.append(row['label'])
    y_pred_p2.append(predict(flower_tree_v2, row))

# Matrisi hesapla
cm_p2 = confusion_matrix(y_true_p2, y_pred_p2, labels=flower_species)

# Görselleştirme
plt.figure(figsize=(12, 10))
sns.heatmap(cm_p2, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=flower_species, yticklabels=flower_species)

plt.title(f'Phase 2 Confusion Matrix\n(Train: 51.25% | Val: {accuracy_v2:.2f}%)', fontsize=14)
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.xticks(rotation=45)

plt.savefig('phase2_confusion_matrix.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Phase 2: Feature Enhancement and Finer Discretization
# 
# **Methodology:**
# To fix the underfitting issue from Phase 1, we added Aspect Ratio to our features. We also increased the discretization from 3 to 7 bins so we wouldn't lose too much detail in the data.
# 
# **Performance Reporting & Analysis:**
# Even with the new feature and more bins, the validation accuracy barely improved. Looking at the F1-Scores, the ID3 tree is still getting confused by flowers that share similar colors. Basically, the underfitting problem is still there.

# %%
y_true_p2 = []
y_pred_p2 = []
for _, row in val_df_discretized_v2.iterrows(): 
    y_true_p2.append(row['label'])
    y_pred_p2.append(predict(flower_tree_v2, row))

print("\n--- PHASE 2: Validation Performance Metrics ---")
calculate_metrics(y_true_p2, y_pred_p2, flower_species)

# %% [markdown]
# ### Phase 2: Experimental Results and Initial Analysis
# In this phase, we tried to improve the model by adding some complexity. We changed discretization to 7 bins to catch finer details in the feature values and added Aspect Ratio as a geometric feature. The goal was to help the ID3 algorithm tell the different flower shapes apart.
# 
# #### 1. Performance Metrics
# Here are the results after making these changes:
# * **Training Accuracy:** 51.25%
# * **Validation Accuracy:** 29.20%
# * **Bin Count:** 7 (Level 0 - Level 6)
# * **New Feature:** Aspect Ratio (Width/Height)
# 
# #### 2. Analysis of the Results (Underfitting)
# Looking at the numbers, there is roughly a 22% gap between the training and validation accuracy. However, the biggest issue is that the training accuracy is stuck at 51.25%. For a decision tree trained on 6,000 images, a 51% training score is a clear sign of underfitting. This happens because our current features (Mean Hue, Saturation, Value, and Edge Density) just aren't descriptive enough. Even with 7 bins, different flower species still look too similar to tree. For instance, a red rose and a red tulip can have the exact same average color, leaving the ID3 algorithm without a clear way to split them.
# 
# #### 3. Confusion Matrix Observations
# When we check the confusion matrix, there is obvious confusion between classes with similar color profiles. Diagonal line is still weak, and many images end up in the wrong categories. This shows that relying on a "Mean Color" approach fails to capture the actual texture or the specific color variations inside petals.
# 
# #### 4. Conclusion and Next Steps
# Overall, Phase 2 proved that simply adding more bins or a basic shape feature isn't enough to pass the 30% accuracy mark. To solve this underfitting issue, the model needs deeper information. 
# 
# In the next step (Phase 3), we plan to add Color Standard Deviation to capture texture patterns, along with raw RGB channel means to give the tree more specific data to work with. Our main goal is to push the training accuracy past 70% and then use pruning to optimize the final tree.

# %%
def extract_features_v3(image_path):
    """
    Final feature set including color variance and raw RGB intensities.
    Designed to maximize Information Gain for the ID3 algorithm.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 1. Geometric Feature
    h, w, _ = img.shape
    aspect_ratio = w / h
    
    # Preprocessing
    img_resized = cv2.resize(img, (100, 100))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # 2. Color Variance (Standard Deviation)
    # Essential for distinguishing textured petals from solid ones
    std_hue = np.std(hsv[:, :, 0])
    std_sat = np.std(hsv[:, :, 1])
    
    # 3. Mean RGB Channels
    # Provides raw color intensity that HSV sometimes averages out
    mean_rgb = np.mean(img_resized, axis=(0, 1)) # B, G, R
    
    # 4. Phase 1 & 2 Features
    mean_hue = np.mean(hsv[:, :, 0])
    mean_sat = np.mean(hsv[:, :, 1])
    edges = cv2.Canny(img_resized, 100, 200)
    edge_density = np.sum(edges > 0) / 10000
    
    return {
        'mean_hue': mean_hue,
        'std_hue': std_hue,
        'mean_saturation': mean_sat,
        'std_saturation': std_sat,
        'mean_red': mean_rgb[2],
        'mean_green': mean_rgb[1],
        'edge_density': edge_density,
        'aspect_ratio': aspect_ratio
    }

# %%
def create_dataset_v3(source_path):
    """
    Iterates through folders to build the high-dimensional Phase 3 dataframe.
    """
    data_list = []
    categories = sorted(os.listdir(source_path))
    
    print(f"--- Starting Phase 3 Feature Extraction: {source_path} ---")
    
    for category in categories:
        cat_path = os.path.join(source_path, category)
        if not os.path.isdir(cat_path):
            continue
            
        print(f"Analyzing Species: {category}...")
        for img_name in os.listdir(cat_path):
            full_path = os.path.join(cat_path, img_name)
            features = extract_features_v3(full_path)
            if features:
                features['label'] = category
                data_list.append(features)
                
    return pd.DataFrame(data_list)

# Execution (Processing 6,000 images)
train_df_v3 = create_dataset_v3(TRAIN_PATH)
val_df_v3 = create_dataset_v3(VAL_PATH)

# %%
# Discretize using our robust V2 function
train_df_dis_v3 = discretize_v2(train_df_v3, bin_count=7)
val_df_dis_v3 = discretize_v2(val_df_v3, bin_count=7)

# Feature list for the final model
features_final = ['mean_hue', 'std_hue', 'mean_saturation', 'std_saturation', 
                  'mean_red', 'mean_green', 'edge_density', 'aspect_ratio']

# Build the tree
print("Building the Phase 3 Tree... This may take a moment due to complexity.")
flower_tree_final = build_id3_tree(train_df_dis_v3, train_df_dis_v3, features_final)

# Evaluate
train_acc_final = calculate_accuracy(flower_tree_final, train_df_dis_v3)
val_acc_final = calculate_accuracy(flower_tree_final, val_df_dis_v3)

print("\n" + "="*50)
print(f"PHASE 3 RESULTS (Before Pruning)")
print(f"Training Accuracy:   {train_acc_final:.2f}%")
print(f"Validation Accuracy: {val_acc_final:.2f}%")
print("="*50)

# %% [markdown]
# ### Phase 3: High-Dimensional Feature Extraction
# 
# **Methodology:**
# To get better Information Gain, we added Color Variance and Raw RGB Channel Means to our features.
# 
# **Performance Reporting & Analysis:**
# The tree basically learned the training data perfectly, reaching a 97.63% training accuracy. But our validation accuracy dropped down to 26.40%. This is a clear case of **Overfitting**. The tree grew way too deep and just memorized the specific details of the training images instead of finding general rules. Because of this, the precision, recall, and F1-scores ended up being very inconsistent.

# %%
y_true_p3 = []
y_pred_p3 = []
for _, row in val_df_dis_v3.iterrows(): 
    y_true_p3.append(row['label'])
    y_pred_p3.append(predict(flower_tree_final, row))

print("\n--- PHASE 3: Validation Performance Metrics (Overfitted) ---")
calculate_metrics(y_true_p3, y_pred_p3, flower_species)

# %% [markdown]
# In this phase, we wanted to see if the ID3 algorithm could handle the dataset better if we provided more complex features. After adding Standard Deviation and the RGB channels, the training accuracy jumped to 97.63%. This shows the algorithm is working and successfully built a very deep tree.
# 
# However, the validation accuracy dropped to 26.40%. The model is obviously overfitting. It grew too deep and just memorized the training images instead of actually learning the general flower patterns. To fix this, we have to simplify the tree.
# 
# ### Phase 4: Post-Pruning
# Now we will apply our post-pruning logic. The code will go through this massive tree and cut out the unnecessary branches that are causing mistakes on the validation set.

# %%
def post_prune_tree(tree, train_df, val_df, target_col='label'):
    """
    Bottom-up pruning with safety checks for empty subsets.
    """
    # If it's already a leaf, no prune
    if not isinstance(tree, dict):
        return tree

    feature = list(tree.keys())[0]
    branches = tree[feature]

    # 1. Recursively prune subtrees
    for value, subset in branches.items():
        if isinstance(subset, dict):
            train_subset = train_df[train_df[feature] == value]
            val_subset = val_df[val_df[feature] == value]
            tree[feature][value] = post_prune_tree(subset, train_subset, val_subset, target_col)

    # 2. Safety Check
    if len(val_df) == 0:
        return tree

    # 3. Accuracy Before vs After
    acc_before = calculate_accuracy(tree, val_df)
    
    # Get majority class from TRAINING data
    if len(train_df) > 0:
        majority_class = train_df[target_col].mode()[0]
    else:
        return tree 

    # Test accuracy if we replaced this whole node with a single leaf
    correct_if_pruned = (val_df[target_col] == majority_class).sum()
    acc_after = (correct_if_pruned / len(val_df)) * 100

    # Decision: If pruning doesn't hurt performance on validation data, simplify it!
    if acc_after >= acc_before:
        return majority_class
    
    return tree

# --- EXECUTION ---
print("Applying Post-Pruning with ZeroDivision protection...")
final_pruned_tree = post_prune_tree(copy.deepcopy(flower_tree_final), train_df_dis_v3, val_df_dis_v3)

# Final Metrics
final_train_acc = calculate_accuracy(final_pruned_tree, train_df_dis_v3)
final_val_acc = calculate_accuracy(final_pruned_tree, val_df_dis_v3)

print("\n" + "="*50)
print(f"FINAL OPTIMIZED RESULTS")
print(f"Final Training Accuracy:   {final_train_acc:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
print("="*50)

# %%
def generate_final_report_matrix(tree, df, classes):
    """
    Self-contained function to plot the final pruned results.
    """
    y_true = []
    y_pred = []
    
    print("Generating final predictions...")
    for _, row in df.iterrows():
        y_true.append(row['label'])
        # Calling existing predict function
        y_pred.append(predict(tree, row))
    
    # Calculate Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'Final Optimized Confusion Matrix\n(Validation Accuracy: 48.80%)', fontsize=14)
    plt.ylabel('Actual Species')
    plt.xlabel('Predicted Species')
    plt.xticks(rotation=45)
    
    # Save file for report
    plt.savefig('final_optimized_matrix.png', bbox_inches='tight')
    plt.show()
    print("Final matrix saved as 'final_optimized_matrix.png'")

#RUN
generate_final_report_matrix(final_pruned_tree, val_df_dis_v3, flower_species)

# %% [markdown]
# ### Phase 4: Model Optimization (Post-Pruning)
# 
# **Methodology:**
# To fix the overfitting issue, we used a bottom-up pruning approach. 
# 
# **Performance Reporting & Analysis:**
# The pruning process actually helped the model generalize much better. The precision, recall, and F1-scores are finally stable and look better than what we had in Phase 3. Overall, this pruned tree is our best model version.

# %%
y_true_p4 = []
y_pred_p4 = []
for _, row in val_df_dis_v3.iterrows():
    y_true_p4.append(row['label'])
    y_pred_p4.append(predict(final_pruned_tree, row))

print("\n--- PHASE 4: Validation Performance Metrics (Pruned Model) ---")
calculate_metrics(y_true_p4, y_pred_p4, flower_species)

# %%
def calculate_metrics_from_cm(cm, classes):
    num_classes = len(classes)
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    total_samples = np.sum(cm)
    overall_accuracy = np.trace(cm) / total_samples
    
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print("-" * 80)
    print(f"{'Class':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 80)
    
    for i in range(num_classes):
        #TP, FP, FN, TN calculation
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = total_samples - TP - FP - FN
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) if total_samples > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        print(f"{classes[i]:<15} | {accuracy:.4f}     | {precision:.4f}     | {recall:.4f}    | {f1:.4f}")
        
    print("-" * 80)
    print(f"{'Macro Average':<15} | {np.mean(accuracies):.4f}     | {np.mean(precisions):.4f}     | {np.mean(recalls):.4f}    | {np.mean(f1_scores):.4f}")
    print("-" * 80)

# --- 1. EVALUATING ON TEST SET ---
print("--- PHASE 5: TEST SET EVALUATION ---")
test_df_raw = create_dataset_v3(TEST_PATH)
test_df_discretized = discretize_v2(test_df_raw, bin_count=7)

y_true_test = []
y_pred_test = []
for _, row in test_df_discretized.iterrows():
    y_true_test.append(row['label'])
    y_pred_test.append(predict(final_pruned_tree, row))

print("\n--- Test Set Classification Performance Metrics ---")
calculate_metrics(y_true_test, y_pred_test, flower_species)

print("\n--- DECISION TREE RULES (Root-to-Leaf Paths) ---")
def print_tree_rules(tree, rule=""):
    if not isinstance(tree, dict):
        print(f"IF {rule} THEN Predict = {tree}")
        return
    feature = list(tree.keys())[0]
    for value, subtree in tree[feature].items():
        new_rule = f"{rule} AND {feature} == '{value}'" if rule else f"{feature} == '{value}'"
        print_tree_rules(subtree, new_rule)

print("Rules for the Best Model (Final Pruned Tree):")
print_tree_rules(final_pruned_tree)

# %% [markdown]
# ### Phase 5: Test Set Results and Final Observations
# The model hit a final test accuracy of 39.61%. It's a bit lower than our validation accuracy, but honestly, this is what we expected when using a basic ID3 tree with manually extracted image features for 10 different flower types.
# 
# Looking at the results, the tree did really well on water lilies (F1-Score: 0.95) and sunflowers (F1-Score: 0.53). This makes sense since they have very distinct colors and edge patterns. However, classes like tulip and carnation scored much lower. Their red and pink colors, along with similar petal edges, just overlap too much and confuse the model.
# 
# Finally, we printed out the root-to-leaf decision rules. The huge number of rules just shows how complicated the tree had to get to separate these images using our simple ID3 algorithm.


