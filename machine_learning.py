import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# --- Chargement et préparation des données ---

# Options d'affichage
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Chargement
df = pd.read_csv('Customer_DF.csv', index_col=0)

# Copie de travail
df_processed = df.copy()

# Création de nouvelles features
df_processed['payment_transaction_ratio'] = np.where(
    df_processed['No_Transactions'] > 0,
    df_processed['No_Payments'] / df_processed['No_Transactions'], 0)

df_processed['order_transaction_ratio'] = np.where(
    df_processed['No_Transactions'] > 0,
    df_processed['No_Orders'] / df_processed['No_Transactions'], 0)

df_processed['total_activity'] = (
    df_processed['No_Transactions'] + df_processed['No_Orders'] + df_processed['No_Payments'])

df_processed['is_inactive'] = (df_processed['No_Transactions'] == 0).astype(int)
df_processed['has_mismatch'] = (df_processed['No_Transactions'] != df_processed['No_Orders']).astype(int)

# Sélection des features
feature_columns = [
    'No_Transactions', 'No_Orders', 'No_Payments',
    'payment_transaction_ratio', 'order_transaction_ratio',
    'total_activity', 'is_inactive', 'has_mismatch'
]

X = df_processed[feature_columns]
y = df_processed['Fraud']

# Séparation train/test stratifiée
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Optimisation du modèle Random Forest ---

rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_

# --- Évaluation finale ---

y_pred = best_rf.predict(X_test_scaled)
y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

# Affichage du rapport de classification
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matrice de Confusion - Random Forest Optimisé')
plt.xlabel('Prédictions')
plt.ylabel('Réel')
plt.show()

# ROC AUC
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC sur le test set : {roc_auc:.4f}")
