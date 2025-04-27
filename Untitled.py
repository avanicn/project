#!/usr/bin/env python
# coding: utf-8
import pandas as pd

# Load the dataset
df = pd.read_csv("CICIDS2017_sample.csv")

# Show the first few rows
df.head()

# In[3]:


# Check the column names and number of missing values
df.info() 


# In[5]:


# See which columns might be all 0 or not helpful
df.describe().T


# In[7]:


# See what labels or attack types we have
df['Label'].value_counts()



# In[9]:


# Convert labels to binary: 0 for BENIGN, 1 for any attack
df['Attack'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)


# In[11]:


# Drop non-numeric and unnecessary columns
df = df.drop(['Label'], axis=1)

# Drop any columns with non-numeric data (just in case)
df = df.select_dtypes(include='number')

# Split into features and target
X = df.drop('Attack', axis=1)
y = df['Attack']


# In[17]:


import numpy as np

# Replace inf/-inf with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values
X.dropna(inplace=True)

# Make sure y matches the updated X
y = y[X.index]


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# See results
print(classification_report(y_test, y_pred))


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[23]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(scores)
print("Average accuracy:", scores.mean())


# In[24]:


print(classification_report(y_test, y_pred))


# In[27]:


print("Done. Report generated.")
print(classification_report(y_test, y_pred))


# In[29]:


import pandas as pd

# Convert to DataFrame (if not already)
df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)

# Check for any overlap
common = pd.merge(df_train, df_test, how='inner')
print(f"Number of duplicate rows between train and test: {len(common)}")


# In[31]:


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# STEP 1: Clean the full dataset FIRST
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)

# Align target (in case some rows dropped from X)
y = y.loc[X.index]

# STEP 2: Now do the train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)


# In[33]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[35]:


y.value_counts(normalize=True)


# In[37]:


train_unique = X_train.drop_duplicates()
test_unique = X_test[~X_test.apply(tuple, axis=1).isin(train_unique.apply(tuple, axis=1))]


# In[41]:


print(X_clean.columns)


# In[43]:


from sklearn.model_selection import train_test_split

X_clean = df.drop_duplicates()
X = X_clean.drop('Attack', axis=1)
y = X_clean['Attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)


# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)


# In[47]:


print(X_clean.columns)


# In[49]:


# Drop duplicates first
X_clean = df.drop_duplicates()

# Separate features and label (corrected column name)
X = X_clean.drop('Attack', axis=1)
y = X_clean['Attack']

# Clean data: remove infinities and NaNs
import numpy as np
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)

# Align y with cleaned X
y = y.loc[X.index]

# Split into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)


# In[ ]:





# In[51]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)


# In[53]:


# Show precision, recall, F1-score, accuracy
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Show confusion matrix
print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[55]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Benign", "Attack"], cmap='Blues')


# In[49]:


# Make predictions using your trained model
y_pred = model.predict(X_test)


# In[51]:


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Visual confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Benign", "Attack"],
    cmap='Blues',
    colorbar=True
)

# Add labels and display
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(False)
plt.show()


# In[53]:


get_ipython().system('pip install shap')


# In[55]:


import shap

# Create explainer for your trained tree-based model (e.g. RandomForest)
explainer = shap.TreeExplainer(model)


# In[57]:


# Choose a small slice of your test set to keep it fast
X_sample = X_test[:100]


# In[59]:


# Get SHAP values for the sample
shap_values = explainer.shap_values(X_sample)


# In[61]:


# For binary classification, use shap_values[1] for the "Attack" class
shap.summary_plot(shap_values[1], X_sample, plot_type="dot")


# In[63]:


print("X_sample shape:", X_sample.shape)
print("shap_values[1] shape:", shap_values[1].shape)


# In[65]:


# Transpose SHAP values to match shape (100, 77)
shap_fixed = shap_values[1].T

# Plot it!
shap.summary_plot(shap_fixed, X_sample, plot_type="dot")


# In[67]:


print("X_sample shape:", X_sample.shape)          # should be (100, 77)
print("shap_values[1] shape:", shap_values[1].shape)  # currently (77, 2)


# In[69]:


# Fix the SHAP shape to match X_sample
shap_fixed = shap_values[1].T  # makes it (2, 77)

print("shap_fixed shape:", shap_fixed.shape)


# In[71]:


import shap

# Use a tiny sample to keep SHAP fast and safe
X_sample = X_test[:10]  # Keep it very small to avoid shape bugs

# Recalculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Print shape sanity check
print("X_sample shape:", X_sample.shape)
print("shap_values[1] shape:", shap_values[1].shape)

# Should now match: (10, number of features)
# Plot the summary
shap.summary_plot(shap_values[1], X_sample, plot_type="dot")


# In[73]:


import shap

# Use a small sample for speed
X_sample = X_test[:10]

# Use the new SHAP explainer that handles this stuff better
explainer = shap.Explainer(model, X_sample)

# Compute SHAP values
shap_values = explainer(X_sample)

# Summary plot (will just work!)
shap.summary_plot(shap_values, X_sample)


# In[75]:


import shap

# Sample just 100 rows to speed up computation
X_sample = X_test.sample(100, random_state=42)

# SHAP for tree models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# For binary classification: shap_values[1] is class "Attack"
print(f"SHAP values shape: {shap_values[1].shape}")
print(f"X_sample shape: {X_sample.shape}")

# Plot the summary
shap.summary_plot(shap_values[1], X_sample)


# In[81]:


# Transpose from (features, classes) to (samples, features)
# Assuming you're using something like a LinearExplainer or kernel-based model

# Re-run SHAP value computation
explainer = shap.Explainer(model, X_sample)  # Automatically selects right explainer
shap_values = explainer(X_sample)  # NEW API returns an object with shap_values.values

# Print shapes to debug
print("SHAP values shape:", shap_values.values.shape)  # Should be (100, 77)
print("X_sample shape:", X_sample.shape)

# Plot with the new object (shap.Explanation)
shap.summary_plot(shap_values[..., 1], X_sample)



# In[79]:


shap.summary_plot(shap_values[..., 1], X_sample)


# In[83]:


#Benchmarking

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Define models to benchmark
models = {
    "Random Forest": model,  # your trained RF model
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}

# Evaluate each model
for name, clf in models.items():
    # Skip retraining Random Forest
    if name != "Random Forest":
        clf.fit(X_train, y_train)

    y_pred_bench = clf.predict(X_test)
    print(f"\n📊 {name} Performance:")
    print(classification_report(y_test, y_pred_bench))


# In[86]:


import joblib

joblib.dump(model, 'random_forest_model.pkl')


# In[ ]:




