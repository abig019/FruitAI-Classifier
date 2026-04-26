import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore
import pickle
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# ── LOAD ──────────────────────────────────────────────────
df = pd.read_csv('fruit_data.csv')
print("✅ Data loaded:", df.shape)
print(df['quality'].value_counts())
print()

# ── ENCODE TARGET ─────────────────────────────────────────
le = LabelEncoder()
df['quality'] = le.fit_transform(df['quality'])
print("Encoding:", dict(zip(le.classes_, le.transform(le.classes_))))
print()

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# ── FEATURES AND TARGET ───────────────────────────────────
X = df.drop('quality', axis=1)
y = df['quality']
feature_names = X.columns.tolist()
print("Features:", feature_names)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# ── SPLIT ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
# WHY stratify=y? Ensures same Fresh/Ripe/Rotten ratio
# in both train and test sets — NEW concept vs Project 1!
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
print()

# ── TRAIN DECISION TREE ───────────────────────────────────
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
# WHY max_depth=5? Prevents overfitting — tree won't memorize data
dt.fit(X_train, y_train) # type: ignore
dt_acc = accuracy_score(y_test, dt.predict(X_test)) # type: ignore
print(f"Decision Tree Accuracy : {dt_acc:.4f}")

# ── TRAIN RANDOM FOREST ───────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train) # type: ignore
rf_acc = accuracy_score(y_test, rf.predict(X_test)) # type: ignore
print(f"Random Forest Accuracy : {rf_acc:.4f}")
print()

# ── PICK BEST ─────────────────────────────────────────────
if dt_acc >= rf_acc:
    best_model, best_pred = dt, dt.predict(X_test) # type: ignore
    best_name = "Decision Tree"
else:
    best_model, best_pred = rf, rf.predict(X_test) # type: ignore
    best_name = "Random Forest"

print(f"Best model: {best_name}")
print()

# ── CLASSIFICATION REPORT ─────────────────────────────────
print("── Classification Report ───────────")
print(classification_report(y_test, best_pred,
      target_names=le.classes_,
      zero_division=0))
# Shows precision/recall/f1 per fruit type — NEW vs Project 1!

# ── CONFUSION MATRIX ──────────────────────────────────────
cm = confusion_matrix(y_test, best_pred) # type: ignore
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# ── SAVE ──────────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('model_name.pkl', 'wb') as f:
    pickle.dump(best_name, f)

print(f"✅ Model saved — {best_name}")