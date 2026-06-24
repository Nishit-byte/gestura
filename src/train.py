
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs('models', exist_ok=True)

print("Loading data...")
df = pd.read_csv('data/gestures.csv')
print(f"Total samples: {len(df)}")
print(f"Gesture distribution:\n{df['label'].value_counts()}\n")

has_sessions = 'session_id' in df.columns
if has_sessions:
    n_sessions = df['session_id'].nunique()
    sessions_per_class = df.groupby('label')['session_id'].nunique()
    min_sessions_per_class = sessions_per_class.min()
    print(f"Found {n_sessions} distinct recording sessions for grouped CV")
    print(f"Sessions per gesture:\n{sessions_per_class}\n")
else:
    print("No session_id column found — falling back to regular K-Fold")
    print("(re-record with the updated collect_data.py for a more honest CV score)\n")
    min_sessions_per_class = 0

feature_cols = [c for c in df.columns if c.startswith('f')]
X = df[feature_cols].values
y = df['label'].values
groups = df['session_id'].values if has_sessions else None

le = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"Classes: {list(le.classes_)}\n")

# ── Train/test split for the classification report ─────────────────────────
# (still randomly shuffled here — this is fine for a quick report, the
#  rigorous check is the grouped CV below)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("=== Classification Report (random split, can be optimistic) ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── The honest accuracy estimate ────────────────────────────────────────────
if has_sessions and min_sessions_per_class >= 2:
    n_splits = min(5, min_sessions_per_class)
    gkf = GroupKFold(n_splits=n_splits)
    cv_scores = cross_val_score(model, X, y_enc, cv=gkf, groups=groups)
    print(f"\n=== Grouped {n_splits}-fold CV accuracy (honest estimate) ===")
    print(f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("Individual fold scores:", [f"{s:.3f}" for s in cv_scores])
elif has_sessions:
    print(f"\n=== Grouped CV skipped ===")
    print(f"Every gesture currently has only {min_sessions_per_class} recording session(s).")
    print("Grouped CV needs at least 2 separate sessions per gesture to work —")
    print("otherwise a fold ends up with zero examples of some gesture in training.")
    print("\nTo fix: run collect_data.py again and re-record EACH gesture in a")
    print("SEPARATE pass (e.g. press 1 for open_hand, stop, later press 1 again")
    print("for a second open_hand session at a different time/distance/lighting).")
    print("\nFalling back to regular 5-fold CV for now (may be optimistic):")
    cv_scores = cross_val_score(model, X, y_enc, cv=5)
    print(f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
else:
    cv_scores = cross_val_score(model, X, y_enc, cv=5)
    print(f"\n=== Regular 5-fold CV accuracy (may be optimistic) ===")
    print(f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

joblib.dump(model, 'models/gesture_model.pkl')
joblib.dump(le,    'models/label_encoder.pkl')
print("\nSaved → models/gesture_model.pkl")
print("Saved → models/label_encoder.pkl")
print("\nDone. Now run: python src/app.py")