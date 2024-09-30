import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from scipy.stats import norm

df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

df.shape

df.head()

df.info()

df.isnull().sum()

df.describe()

print('Number of fraudulent transactions \t: {}'.format(df['isFraud'].sum()))
print('Number of non-fraudulent transactions \t: {}'.format(len(df[df['isFraud']==0])))
print('\nNumber of transactions flagged fraud \t: {}'.format(len(df[df['isFlaggedFraud']==1])))
print('Number of transactions flagged non-fraud: {}'.format(len(df[df['isFlaggedFraud']==0])))

X = df[df['nameDest'].str.contains('M')]
X.head()

fraud_ratio = df['isFraud'].value_counts()/len(df)
flaggedFraud_ratio =  df['isFlaggedFraud'].value_counts()/len(df)

print(f'Fraud ratio \n{fraud_ratio} \n\nFlagged fraud ratio \n{flaggedFraud_ratio}')

df['type'].unique()

fraudby_type = df.groupby(['type', 'isFraud']).size().unstack(fill_value=0)
flaggedFraudby_type = df[df['isFlaggedFraud']==1].groupby('type')['isFlaggedFraud'].count()

print(f'Fraud per transaction type: \n{fraudby_type}\n \nFlagged fraud per transaction type: \n{flaggedFraudby_type}')

df[df['isFraud']==1].describe()

len(df[(df['amount'] == df['oldbalanceOrg'])])

len(df[(df['amount'] == df['oldbalanceOrg']) & (df['isFraud'] == 1)] )

df_outliers = df[(df['amount'] != df['oldbalanceOrg']) & (df['isFraud'] == 1)]

df_outliers.groupby('type')['type'].count()

df_outliers[df_outliers['type'] == 'CASH_OUT'].describe()

df_outliers[df_outliers['type'] == 'TRANSFER'].describe()

df[df['isFlaggedFraud'] == 1].describe()

len(df[  (df['oldbalanceOrg'] == df['newbalanceOrig']) 
       & (df['oldbalanceDest'] == df['newbalanceDest']) 
       & (df['amount']>200.000) 
       & (df['type']=='TRANSFER') ])

len(df[df['isFlaggedFraud'] == 1])

dff_outliers = df[  (df['oldbalanceOrg'] == df['newbalanceOrig']) 
       & (df['oldbalanceDest'] == df['newbalanceDest']) 
       & (df['amount']>200.000) 
       & (df['type']=='TRANSFER') 
       & (df['isFlaggedFraud']==0)]
dff_outliers.describe()

df[df['nameOrig'].str.startswith('M')].describe()

df[df['nameDest'].str.startswith('M')].describe()

df_missing = df[df['nameDest'].str.startswith('M')]

df_missing['type'].unique()

import numpy as np
df.loc[df['nameDest'].str.startswith('M'), ['oldbalanceDest']] = np.NaN
print('{} rows updated with NaN'.format(df['oldbalanceDest'].isnull().sum()))

df=df.interpolate()

df[df['oldbalanceDest'].isnull()]

df.loc[df['oldbalanceDest'].isnull(), 'oldbalanceDest'] = 0

df.isnull().values.any()

df[(df['type']=='PAYMENT') & (df.nameDest.str.get(0) != 'M')]

newbalanceDest = df.loc[df.nameDest.str.get(0) == 'M', 'oldbalanceDest'] + df.loc[df.nameDest.str.get(0) == 'M','amount']

df.loc[df['nameDest'].str.get(0) == 'M', ['newbalanceDest']] = newbalanceDest

len(df[(df['nameDest'].str.get(0) == 'M') & (df['amount'] == df['oldbalanceOrg'])])

df[df['nameDest'].str.get(0) == 'M'].describe()

df.describe()

df = df.drop(['nameOrig', 'nameDest'], axis=1)

df.info()

df.describe()

cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
df['step'] = df['step'] - df['step'].mean() / (df['step'].std())
df[cols] = df[cols].apply(lambda x: (np.log(x+10)))

df.head()

df.describe()

df2 = df[(df['type'].isin(['CASH_OUT', 'TRANSFER']))].copy(deep=True)

df2.info()

df2['step'] = df2['step'] - df2['step'].mean() / (df2['step'].std())

df2.describe()

plt.figure(figsize=(5,10))
labels = ["NotFraud", "Fraud"]
count_classes = df.value_counts(df['isFraud'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()

plt.rcParams['figure.figsize'] =(16, 5)

plt.subplot(1, 2, 1)
sns.distplot(df.step, fit=norm)
plt.title('Frequency of transactions in each step (df dataset)', fontsize = 12)

plt.subplot(1, 2, 2)
sns.distplot(df2.step, fit=norm, color='y')
plt.title('Frequency of transactions in each step (df2 dataset)', fontsize = 12)

plt.show()

plt.rcParams['figure.figsize'] =(14, 12)

plt.subplot(2, 2, 1)
sns.violinplot(x='isFraud',y='step',data=df, palette='Pastel1')
plt.title('Frequency distribution of fraud/step (df dataset)', fontsize = 12)

plt.subplot(2, 2, 2)
sns.violinplot(x='isFlaggedFraud',y='step',data=df, palette='Pastel1')
plt.title('Frequency distribution of flaggedFraud/step (df dataset)', fontsize = 12)

plt.subplot(2, 2, 3)
sns.violinplot(x='isFraud',y='step',data=df2, palette='Pastel2')
plt.title('Frequency distribution of fraud/step (df2 dataset)', fontsize = 12)

plt.subplot(2, 2, 4)
sns.violinplot(x='isFlaggedFraud',y='step',data=df2, palette='Pastel2')
plt.title('Frequency distribution of flaggedFraud/step (df2 dataset)', fontsize = 12)

plt.show()

plt.rcParams['figure.figsize'] =(18, 12)

plt.subplot(2, 2, 1)
sns.violinplot(x='isFraud',y='step',data=df, hue='type', palette='Pastel1')
plt.title('Categorical distribution of fraud/step (df dataset)', fontsize = 12)

plt.subplot(2, 2, 2)
sns.violinplot(x='isFlaggedFraud',y='step',data=df, hue='type', palette='Pastel1')
plt.title('Categorical distribution of flaggedFraud/step (df dataset)', fontsize = 12)

plt.subplot(2, 2, 3)
sns.violinplot(x='isFraud',y='step',data=df2, hue='type', split=True, palette='Pastel2')
plt.title('Categorical distribution of fraud/step (df2 dataset)', fontsize = 12)

plt.subplot(2, 2, 4)
sns.violinplot(x='isFlaggedFraud',y='step',data=df2, hue='type', split=True, palette='Pastel2')
plt.title('Categorical distribution of flaggedFraud/step (df2 dataset)')

plt.show()

plt.rcParams['figure.figsize'] = (18, 12)

plt.subplot(2, 2, 1)
sns.violinplot(x = 'type', y = 'amount', data=df, hue='isFraud', split=True, palette='Pastel1')
plt.title('Categorical distribution of each amount (df dataset)', fontsize = 12)

plt.subplot(2, 2, 2)
sns.violinplot(x = 'type', y = 'amount', data=df, hue='isFlaggedFraud', split=True, palette='Pastel1')
plt.title('Categorical distribution of each amount (df dataset)', fontsize = 12)

plt.subplot(2, 2, 3)
sns.violinplot(x = 'type', y = 'amount', data=df2, hue='isFraud', split=True, palette='Pastel2')
plt.title('Categorical distribution of each amount (df2 dataset)', fontsize = 12)

plt.subplot(2, 2, 4)
sns.violinplot(x = 'type', y = 'amount', data=df2, hue='isFlaggedFraud', split=True, palette='Pastel2')
plt.title('Categorical distribution of each amount (df2 dataset)', fontsize = 12)

plt.show()

def engineer_features(df):
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['balance_ratio_orig'] = df['oldbalanceOrg'] / (df['newbalanceOrig'] + 1)
    df['balance_ratio_dest'] = df['oldbalanceDest'] / (df['newbalanceDest'] + 1)
    return df

data = engineer_features(df)

features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'balance_diff_orig', 'balance_diff_dest', 'balance_ratio_orig', 'balance_ratio_dest', 'type']

X = data[features]
y = data['isFraud']

X = pd.get_dummies(X, columns=['type'], prefix='type')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), early_stopping_rounds=100)

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'catboost_model.joblib')
joblib.dump(scaler, 'standard_scaler.joblib')
joblib.dump(X.columns.tolist(), 'feature_names.joblib')

print("Model, scaler, and feature names saved successfully.")