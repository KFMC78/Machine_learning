import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the data
df = pd.read_csv('Customer_DF.csv', index_col=0)

# Create a working copy
df_processed = df.copy()

# Create new features
df_processed['payment_transaction_ratio'] = np.where(df_processed['No_Transactions'] > 0,
                                                     df_processed['No_Payments'] / df_processed['No_Transactions'],
                                                     0)

df_processed['order_transaction_ratio'] = np.where(df_processed['No_Transactions'] > 0,
                                                   df_processed['No_Orders'] / df_processed['No_Transactions'],
                                                   0)

df_processed['total_activity'] = df_processed['No_Transactions'] + df_processed['No_Orders'] + df_processed['No_Payments']

# Binary flags
df_processed['is_inactive'] = (df_processed['No_Transactions'] == 0).astype(int)
df_processed['has_mismatch'] = (df_processed['No_Transactions'] != df_processed['No_Orders']).astype(int)

# Select features for modeling
feature_columns = ['No_Transactions', 'No_Orders', 'No_Payments',
                   'payment_transaction_ratio', 'order_transaction_ratio',
                   'total_activity', 'is_inactive', 'has_mismatch']

X = df_processed[feature_columns]
y = df_processed['Fraud']

# Split the data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to dataframe for easier manipulation
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
