# =============================================================================
# PROJET DÉTECTION DE FRAUDE 
# Équipe : ADYEL Ilyès, ATTIOGBE Killian, DIAW Ismaël
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# 1. PREPARATION DES DONNEES (DATA ENGINEERING)
# =============================================================================
print("--- 1. Chargement et Feature Engineering ---")

try:
    df = pd.read_csv('Customer_DF.csv')
except FileNotFoundError:
    print("ERREUR : Fichier introuvable.")
    raise

df_processed = df.copy()


df_processed['payment_per_transaction'] = df_processed.apply(
    lambda x: x['No_Payments'] / x['No_Transactions'] if x['No_Transactions'] > 0 else 0, axis=1
)
df_processed['order_per_transaction'] = df_processed.apply(
    lambda x: x['No_Orders'] / x['No_Transactions'] if x['No_Transactions'] > 0 else 0, axis=1
)

df_processed['total_activity'] = df_processed['No_Transactions'] + df_processed['No_Orders'] + df_processed['No_Payments']
df_processed['avg_activity'] = df_processed['total_activity'] / 3

df_processed['is_inactive'] = (df_processed['No_Transactions'] == 0).astype(int)
df_processed['transaction_order_mismatch'] = (df_processed['No_Transactions'] != df_processed['No_Orders']).astype(int)
df_processed['has_high_payment_ratio'] = (df_processed['payment_per_transaction'] > 1).astype(int)

df_processed['payment_anomaly'] = np.abs(df_processed['No_Payments'] - df_processed['No_Transactions'])
df_processed['order_anomaly'] = np.abs(df_processed['No_Orders'] - df_processed['No_Transactions'])

# Sélection des colonnes
exclude_columns = ['customerEmail', 'customerPhone', 'customerDevice', 
                  'customerIPAddress', 'customerBillingAddress', 'Fraud']
feature_columns = [col for col in df_processed.columns if col not in exclude_columns]

X = df_processed[feature_columns]
y = df_processed['Fraud']

print(f"Features prêtes : {len(feature_columns)}")

# =============================================================================
# 2. EDA (Visuels rapides)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['Fraud'].value_counts().plot(kind='pie', ax=axes[0], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
axes[0].set_title('Fraud Distribution')
axes[0].set_ylabel('')
df['Fraud'].value_counts().plot(kind='bar', ax=axes[1], color=['lightgreen', 'lightcoral'])
axes[1].set_title('Fraud Count')
plt.show()

# =============================================================================
# 3. PARTIE ISMAËL : FEATURE IMPORTANCE 
# =============================================================================
print("\n--- 3. Analyse Feature Importance (Ismaël) ---")


X_train_feat, X_test_feat, y_train_feat, y_test_feat = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler_feat = StandardScaler()
X_train_scaled_feat = scaler_feat.fit_transform(X_train_feat)

X_train_scaled_feat = pd.DataFrame(X_train_scaled_feat, columns=feature_columns)


lr_model_feat = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000,
    solver='liblinear'
)
lr_model_feat.fit(X_train_scaled_feat, y_train_feat)


feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': lr_model_feat.coef_[0],
    'abs_coefficient': np.abs(lr_model_feat.coef_[0])
}).sort_values('abs_coefficient', ascending=False)


plt.figure(figsize=(10, 8))
colors = ['red' if x < 0 else 'green' for x in feature_importance['coefficient']]
plt.barh(range(len(feature_importance)), feature_importance['coefficient'], color=colors)
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)


for i, v in enumerate(feature_importance['coefficient']):
    if v < 0:
        plt.text(v - 0.02, i, f'{v:.3f}', ha='right', va='center')
    else:
        plt.text(v + 0.02, i, f'{v:.3f}', ha='left', va='center')

plt.tight_layout()
plt.show()

# Optimisation LogReg 
print("Optimisation LogReg en cours...")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_lr.fit(X_train_scaled_feat, y_train_feat) # On utilise le même split
best_lr_model = grid_search_lr.best_estimator_
print(f"Meilleurs params LogReg : {grid_search_lr.best_params_}")


# =============================================================================
# 4. PARTIE ILYÈS : RANDOM FOREST
# =============================================================================
print("\n--- 4. Optimisation Random Forest (Ilyès) ---")


rf = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}

grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_rf.fit(X_train_scaled_feat, y_train_feat)
best_rf_model = grid_search_rf.best_estimator_
print(f"Meilleurs params RF : {grid_search_rf.best_params_}")


# =============================================================================
# 5. PARTIE KILLIAN : VOTING & VISUALISATION FINALE
# =============================================================================
print("\n--- 5. Voting & Résultats Finaux (Killian) ---")

# Préparation du test set (scaling)
X_test_scaled_feat = scaler_feat.transform(X_test_feat)
X_test_scaled_feat = pd.DataFrame(X_test_scaled_feat, columns=feature_columns)

voting_clf = VotingClassifier(
    estimators=[('lr', best_lr_model), ('rf', best_rf_model)],
    voting='soft'
)
voting_clf.fit(X_train_scaled_feat, y_train_feat)

# ROC Curves
models = [
    {'label': 'LogReg (Ismaël)', 'model': best_lr_model, 'color': 'blue', 'style': '--'},
    {'label': 'Random Forest (Ilyès)', 'model': best_rf_model, 'color': 'green', 'style': '--'},
    {'label': 'Voting (Killian)', 'model': voting_clf, 'color': 'red', 'style': '-'}
]

plt.figure(figsize=(10, 8))
for m in models:
    y_prob = m['model'].predict_proba(X_test_scaled_feat)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_feat, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=m['color'], linestyle=m['style'], lw=2, label=f"{m['label']} (AUC = {roc_auc:.4f})")

plt.plot([0, 1], [0, 1], 'k:')
plt.legend()
plt.title('Comparaison Finale (ROC Curves)')
plt.show()

# Matrice de Confusion
y_pred_final = voting_clf.predict(X_test_scaled_feat)
cm = confusion_matrix(y_test_feat, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion (Voting)')
plt.show()
