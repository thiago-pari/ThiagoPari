#!/usr/bin/env python
# coding: utf-8

# In[131]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from scipy.stats import shapiro, normaltest, mannwhitneyu, pearsonr
from imblearn.over_sampling import SMOTE

# Set the seed for reproducibility
np.random.seed(19259759)

# Load the dataset
data = pd.read_csv('/Users/thiago.pari/Downloads/spotify52kData.csv')


# In[45]:


# Q1
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
for i, feature in enumerate(features):
    row, col = divmod(i, 5)
    sns.histplot(data[feature], bins=30, kde=True, ax=axes[row, col])
    axes[row, col].set_title(feature)
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

normality_results = {}
for feature in features:
    shapiro_test = shapiro(data[feature])
    dagostino_test = normaltest(data[feature])
    normality_results[feature] = {
        'Shapiro-Wilk p-value': shapiro_test.pvalue,
        'D\'Agostino K-squared p-value': dagostino_test.pvalue
    }
print("None of the  none of the distributions are perfectly normal, but valence seems to have the most symmetric distribution around its mean, even though it has some skewness to the left.")


# In[46]:


# Q2
data['duration_minutes'] = data['duration'] / 60000
filtered_data = data[data['duration_minutes'] <= 20]
correlation_full, p_value_full = pearsonr(data['duration_minutes'], data['popularity'])
correlation_filtered, p_value_filtered = pearsonr(filtered_data['duration_minutes'], filtered_data['popularity'])

fig, axs = plt.subplots(1, 2, figsize=(20, 8))
sns.scatterplot(x='duration_minutes', y='popularity', data=data, alpha=0.5, ax=axs[0])
axs[0].set_title('Scatterplot of Song Duration vs Popularity')
axs[0].set_xlabel('Duration (minutes)')
axs[0].set_ylabel('Popularity')
axs[0].annotate(f"Pearson r: {correlation_full:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red')
sns.scatterplot(x='duration_minutes', y='popularity', data=filtered_data, alpha=0.5, ax=axs[1])
axs[1].set_title('Scatterplot of Song Duration vs Popularity (<= 20 minutes)')
axs[1].set_xlabel('Duration (minutes)')
axs[1].set_ylabel('Popularity')
axs[1].annotate(f"Pearson r: {correlation_filtered:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red')
plt.tight_layout()
plt.show()


# In[60]:


# Q3

# Split the data into explicit and non-explicit groups
explicit_songs = data[data['explicit'] == True]['popularity']
non_explicit_songs = data[data['explicit'] == False]['popularity']

# Perform the Mann-Whitney U test
stat, p_value = mannwhitneyu(explicit_songs, non_explicit_songs, alternative='greater')
mean_explicit = explicit_songs.mean()
mean_non_explicit = non_explicit_songs.mean()

# Print the results
print(f"U-Statistic: {stat}")
print(f"P-Value: {p_value}")

# Interpret the results
if p_value < 0.05:
    print("Explicitly rated songs are statistically more popular than non-explicit songs.")
else:
    print("There is no significant difference in popularity between explicit and non-explicit songs.")


# In[61]:


# Q4

# Split the data into major and minor key groups
major_key_songs = data[data['mode'] == 1]['popularity']
minor_key_songs = data[data['mode'] == 0]['popularity']

# Perform the Mann-Whitney U test
stat, p_value = mannwhitneyu(major_key_songs, minor_key_songs, alternative='greater')
mean_major = major_key_songs.mean()
mean_minor = minor_key_songs.mean()

# Print the results
print(f"U-Statistic: {stat}")
print(f"P-Value: {p_value}")

# Interpret the results
if p_value < 0.05:
    print("Songs in major key are statistically more popular than songs in minor key.")
else:
    print("There is no significant difference in popularity between songs in major and minor keys.")


# In[65]:


# Q5

# Create a scatterplot of energy against loudness
plt.figure(figsize=(10, 6))
plt.scatter(data['energy'], data['loudness'], alpha=0.5)
plt.title('Relationship between Energy and Loudness of Songs')
plt.xlabel('Energy')
plt.ylabel('Loudness (dB)')
plt.grid(True)
plt.show()

# Calculate the Pearson correlation coefficient
correlation_coefficient, p_value = stats.pearsonr(data['energy'], data['loudness'])
print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
print(f"P-Value: {p_value}")

# Interpret the results
if p_value < 0.05:
    print("There is a statistically significant correlation between energy and loudness.")
else:
    print("There is no significant correlation between energy and loudness.")


# In[109]:


# Q6
X = data[features]
y = data['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19259759)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

performance_table = pd.DataFrame(columns=['Feature', 'RMSE', 'R²'])
for feature in features:
    X_feature = data[[feature]]
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=19259759)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    performance_table = performance_table.append({'Feature': feature, 'RMSE': rmse, 'R²': r2}, ignore_index=True)

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X, y, test_size=0.2, random_state=19259759)
model_all = LinearRegression()
model_all.fit(X_train_all, y_train_all)
y_pred_all = model_all.predict(X_test_all)
rmse_all = mean_squared_error(y_test_all, y_pred_all, squared=False)
r2_all = r2_score(y_test_all, y_pred_all)
performance_table = performance_table.append({'Feature': 'All features', 'RMSE': rmse_all, 'R²': r2_all}, ignore_index=True)

performance_table


# In[132]:


#Q7
# Define the features and target variable
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = data['popularity']

# Split the data into training and testing sets for all features
X = data[features]
y = data['popularity']
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X, y, test_size=0.2, random_state=19259759)

# Build the model using all features
model_all = LinearRegression()
model_all.fit(X_train_all, y_train_all)
y_pred_all = model_all.predict(X_test_all)
r2_all = r2_score(y_test_all, y_pred_all)

# Build the model using the single feature 'instrumentalness'
X_instrumentalness = data[['instrumentalness']]
X_train_instr, X_test_instr, y_train_instr, y_test_instr = train_test_split(X_instrumentalness, y, test_size=0.2, random_state=19259759)
model_instr = LinearRegression()
model_instr.fit(X_train_instr, y_train_instr)
y_pred_instr = model_instr.predict(X_test_instr)
r2_instr = r2_score(y_test_instr, y_pred_instr)

# Print the R2 values
print(f"R2 value using all features: {r2_all}")
print(f"R2 value using 'instrumentalness' only: {r2_instr}")


# In[81]:


# Q8

# Standardize the features before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initiate PCA
pca = PCA()

# Fit PCA on the standardized data
pca.fit(X_scaled)

# Explained Variance Ratios
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Determine number of components for 95% variance
num_components_95_variance = (cumulative_explained_variance >= 0.95).argmax() + 1

# Plot the explained variance
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

variance_df = pd.DataFrame({
    'Explained Variance': explained_variance_ratio,
    'Cumulative Explained Variance': cumulative_explained_variance
})

variance_df


# In[114]:


#Q9 Can you predict whether a song is in major or minor key from valence? 
# If so, how good is this prediction? If not, is there a better predictor? 
# [Suggestion: It might be nice to show the logistic regression once you are done building the model] 

# Prepare the data for logistic regression
X = data[['valence']]  # Predictor
y = data['mode']       # Dependent variable, assuming 'mode' where 1 = major and 0 = minor

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Fit the model using statsmodels for detailed summary
X_sm = sm.add_constant(X)  # adding a constant for statsmodels
logit_model = sm.Logit(y, X_sm)
result = logit_model.fit()
print(result.summary())

# Define the feature (valence) and target variable (mode)
X_valence = data[['valence']]
y_mode = data['mode']

# Handle class imbalance using stratified sampling
X_train_valence, X_test_valence, y_train_mode, y_test_mode = train_test_split(X_valence, y_mode, test_size=0.2, random_state=19259759, stratify=y_mode)

# Initialize and train the logistic regression model
log_reg_valence = LogisticRegression()
log_reg_valence.fit(X_train_valence, y_train_mode)

# Predict on the test set
y_pred_valence = log_reg_valence.predict(X_test_valence)
y_pred_proba_valence = log_reg_valence.predict_proba(X_test_valence)[:, 1]

# Calculate accuracy and AUC
accuracy_valence = accuracy_score(y_test_mode, y_pred_valence)
auc_valence = roc_auc_score(y_test_mode, y_pred_proba_valence)

# Check if there is a better predictor among the features
better_predictor = None
best_accuracy = accuracy_valence

for feature in features:
    X_feature = data[[feature]]
    X_train_feature, X_test_feature, y_train_mode, y_test_mode = train_test_split(X_feature, y_mode, test_size=0.2, random_state=19259759, stratify=y_mode)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train_feature, y_train_mode)
    
    y_pred_feature = log_reg.predict(X_test_feature)
    accuracy_feature = accuracy_score(y_test_mode, y_pred_feature)
    
    if accuracy_feature > best_accuracy:
        best_accuracy = accuracy_feature
        better_predictor = feature

(accuracy_valence, auc_valence, better_predictor, best_accuracy)


# In[135]:


#Q9 Can you predict whether a song is in major or minor key from valence? 
# If so, how good is this prediction? If not, is there a better predictor?

# Define the feature (valence) and target variable (mode)
X_valence = data[['valence']]
y_mode = data['mode']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=19259759)
X_valence_smote, y_mode_smote = smote.fit_resample(X_valence, y_mode)

# Split the data into training and testing sets
X_train_valence, X_test_valence, y_train_mode, y_test_mode = train_test_split(X_valence_smote, y_mode_smote, test_size=0.2, random_state=19259759)

# Initialize and train the logistic regression model with class weighting
log_reg_valence = LogisticRegression(class_weight='balanced')
log_reg_valence.fit(X_train_valence, y_train_mode)

# Predict on the test set
y_pred_valence = log_reg_valence.predict(X_test_valence)
y_pred_proba_valence = log_reg_valence.predict_proba(X_test_valence)[:, 1]

# Calculate accuracy and AUC
accuracy_valence = accuracy_score(y_test_mode, y_pred_valence)
auc_valence = roc_auc_score(y_test_mode, y_pred_proba_valence)

# Check if there is a better predictor among the features
better_predictor = None
best_accuracy = accuracy_valence
best_auc = auc_valence

for feature in features:
    X_feature = data[[feature]]
    X_feature_smote, y_mode_smote = smote.fit_resample(X_feature, y_mode)
    X_train_feature, X_test_feature, y_train_mode, y_test_mode = train_test_split(X_feature_smote, y_mode_smote, test_size=0.2, random_state=19259759)
    
    log_reg = LogisticRegression(class_weight='balanced')
    log_reg.fit(X_train_feature, y_train_mode)
    
    y_pred_feature = log_reg.predict(X_test_feature)
    y_pred_proba_feature = log_reg.predict_proba(X_test_feature)[:, 1]
    
    accuracy_feature = accuracy_score(y_test_mode, y_pred_feature)
    auc_feature = roc_auc_score(y_test_mode, y_pred_proba_feature)
    
    if accuracy_feature > best_accuracy and auc_feature > best_auc:
        best_accuracy = accuracy_feature
        best_auc = auc_feature
        better_predictor = feature

(accuracy_valence, auc_valence, better_predictor, best_accuracy, best_auc)

print(f"Accuracy of valence: {accuracy_valence}")
print(f"AUC valence: {auc_valence}")
print(f"Better predictor: {better_predictor}")
print(f"Accuracy of better predictor: {best_accuracy}")
print(f"AUC of better predictor: {best_auc}")

# Define the feature (valence) and target variable (mode)
X_valence = data[['valence']]
y_mode = data['mode']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=19259759)
X_valence_smote, y_mode_smote = smote.fit_resample(X_valence, y_mode)

# Split the data into training and testing sets
X_train_valence, X_test_valence, y_train_mode, y_test_mode = train_test_split(X_valence_smote, y_mode_smote, test_size=0.2, random_state=19259759)

# Initialize and train the logistic regression model with class weighting
log_reg_valence = LogisticRegression(class_weight='balanced')
log_reg_valence.fit(X_train_valence, y_train_mode)

# Predict on the test set
y_pred_proba_valence = log_reg_valence.predict_proba(X_test_valence)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_mode, y_pred_proba_valence)
auc_valence = roc_auc_score(y_test_mode, y_pred_proba_valence)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_valence)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Valence')
plt.legend(loc="lower right")
plt.show()


# Define the feature (acousticness) and target variable (mode)
X_acousticness = data[['acousticness']]
y_mode = data['mode']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=19259759)
X_acousticness_smote, y_mode_smote = smote.fit_resample(X_acousticness, y_mode)

# Split the data into training and testing sets
X_train_acousticness, X_test_acousticness, y_train_mode, y_test_mode = train_test_split(X_acousticness_smote, y_mode_smote, test_size=0.2, random_state=19259759)

# Initialize and train the logistic regression model with class weighting
log_reg_acousticness = LogisticRegression(class_weight='balanced')
log_reg_acousticness.fit(X_train_acousticness, y_train_mode)

# Predict on the test set
y_pred_proba_acousticness = log_reg_acousticness.predict_proba(X_test_acousticness)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_mode, y_pred_proba_acousticness)
auc_acousticness = roc_auc_score(y_test_mode, y_pred_proba_acousticness)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_acousticness)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Acousticness')
plt.legend(loc="lower right")
plt.show()


# In[106]:


#Q10 Which is a better predictor of whether a song is classical music – duration or the 
# principal components you extracted in question 8? [Suggestion: You might have to convert
# the qualitative genre label to a binary numerical label (classical or not)] 

# Convert the track_genre to a binary format: 1 if classical, 0 otherwise
data['is_classical'] = (data['track_genre'] == 'classical').astype(int)

# Select the features for PCA
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = data[features]
y = data['is_classical']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA from Question 8
pca = PCA(n_components=6)  # Assuming we decided 6 was adequate from Question 8
X_pca = pca.fit_transform(X_scaled)

# Split the dataset into training and test sets for both original and PCA features
X_train_dur, X_test_dur, y_train, y_test = train_test_split(data[['duration']], y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Logistic regression using duration
model_dur = LogisticRegression()
model_dur.fit(X_train_dur, y_train)
y_pred_dur = model_dur.predict(X_test_dur)

# Logistic regression using PCA components
model_pca = LogisticRegression()
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)

# Evaluate and compare the models
print("Duration Model Accuracy:", accuracy_score(y_test, y_pred_dur))
print("PCA Model Accuracy:", accuracy_score(y_test, y_pred_pca))
print("\nDuration Model Classification Report:")
print(classification_report(y_test, y_pred_dur))
print("PCA Model Classification Report:")
print(classification_report(y_test, y_pred_pca))


# In[105]:


# Extra Credit
time_signature_distribution = data['time_signature'].value_counts().sort_index()
avg_popularity_by_time_signature = data.groupby('time_signature')['popularity'].mean()

plt.figure(figsize=(10, 6))
avg_popularity_by_time_signature.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Popularity by Time Signature')
plt.xlabel('Time Signature (Beats per Measure)')
plt.ylabel('Average Popularity')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

