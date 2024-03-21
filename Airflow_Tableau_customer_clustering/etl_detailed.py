# sql connection
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine, text
# base libraraies 
import pandas as pd
import numpy as np
from datetime import datetime
import datetime
# data viz. libraries
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
# model development, selection libraries 
from scipy.stats import chi2_contingency
from scipy import stats
import warnings
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# step 1. Extact

# connecting to mysql function
def read_credentials(filename):
    credentials = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split(' = ')
            credentials[key] = value.strip("'")
    return credentials


# reading the credentials
credentials = read_credentials('/Users/sajad/Documents/legionella/cred.text')
host_name = credentials['host_name']
db_name = credentials['db_name']
user_name = credentials['user_name']
password_enc = credentials['password_enc']

# establishing the connection, extracating the required columns
engine = create_engine(
    f'mysql+mysqlconnector://{user_name}:{password_enc}@{host_name}/{db_name}')

query = text("""select id, gender, `function`, company, last_bought_product,
                isActive, `status`, last_login, clientSince, language
            from users_db.mock_data""")

df = pd.read_sql(query, engine)


# step 2: transformig the data
# 2.1: preprocessing and wrangling
# 2.1.1 getting to know the data
df.info()
df.describe(include = 'all')

# 2.1.2 changing the data types
id = df['id']
df['last_login'] = pd.to_datetime(df['last_login'])
df['clientSince'] = pd.to_datetime(df['clientSince'])
df['isActive'] = df['isActive'].map({'0': 0, '1': 1})



# 2.2. feature selection
features = df.drop(columns=['id'])


high_dimensions = ['function', 'company', 'last_bought_product', 'language']
for col in high_dimensions:
    features[col].value_counts(dropna=False).head(5).plot.bar()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

features = features.drop(
    columns=['function', 'company', 'last_bought_product', 'language'])  # not used in the final etl


#pearson and cramers v correlations
# def cramerV(label, x):
#     confusion_matrix = pd.crosstab(label, x)
#     chi2 = chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     r, k = confusion_matrix.shape
#     phi2 = chi2/n
#     phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
#     rcorr = r - ((r - 1) ** 2) / (n - 1)
#     kcorr = k - ((k - 1) ** 2) / (n - 1)
#     try:
#         if min((kcorr - 1), (rcorr - 1)) == 0:
#             warnings.warn(
#                 "Unable to calculate Cramer's V using bias correction. Consider not using bias correction", RuntimeWarning)
#             v = 0
#             print("If condition Met: ", v)
#         else:
#             v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
#             print("Else condition Met: ", v)
#     except:
#         print("inside error")
#         v = 0
#     return v


# def plot_cramer(df):
#     cramer = pd.DataFrame(index=df.columns, columns=df.columns)
#     for column_of_interest in df.columns:
#         try:
#             temp = {}

#             columns = df.columns
#             for j in range(0, len(columns)):
#                 v = cramerV(df[column_of_interest], df[columns[j]])
#                 cramer.loc[column_of_interest, columns[j]] = v
#                 if (column_of_interest == columns[j]):
#                     pass
#                 else:
#                     temp[columns[j]] = v
#             cramer.fillna(value=np.nan, inplace=True)
#         except:
#             print('Dropping row:', column_of_interest)
#             pass
#     plt.figure(figsize=(7, 7))
#     heatmap = sns.heatmap(cramer, annot=True, fmt='.2f')
#     heatmap.set_xticklabels(heatmap.get_xticklabels(),
#                             rotation=45, horizontalalignment='right')

#     plt.title(
#         y=1.03, label="Cross Correlation plot on features with Cramer's and Pearson Correlation Values")
#     plt.show()


# plot_cramer(features)


# 3.3. feature engineering

# 3.3.1 handling missing values

# looking at the null values pattern
sns.heatmap(features.isnull(), cbar=False)
plt.xticks(rotation=45, ha='right')
plt.show()

sns.heatmap(features.isnull().corr(), cbar=False, annot=True)
plt.xticks(rotation=45, ha='right')
plt.show()

msno.heatmap(features)
plt.show()

# features.dropna().shape


# imputing missing values
# last loging with knn

features['days_last_login'] = (
    datetime.datetime.today() - features['last_login']).dt.days
features = features.drop(columns=['last_login'])

features['days_clientSince'] = (
    datetime.datetime.today() - features['clientSince']).dt.days
features = features.drop(columns=['clientSince'])

# features.info()

imputer = KNNImputer(n_neighbors=5, weights="uniform")
imputed_data = imputer.fit_transform(features[['days_last_login']])
features.loc[:, 'days_last_login'] = imputed_data
# features['days_last_login'].value_counts(dropna=False)

# status and gender with mode
imputer = SimpleImputer(strategy='most_frequent', missing_values=None)
imputed_data = imputer.fit_transform(features[['gender', 'status']])
imputed_df = pd.DataFrame(imputed_data, columns=['gender', 'status'])
features[['gender', 'status']] = imputed_df

# 3.2 standardizing the days last login and days_clientSince columns
scaler = MinMaxScaler(feature_range=(0, 1))
days_data = features[['days_last_login', 'days_clientSince']]
scaled_data = scaler.fit_transform(days_data)
features = features.drop(columns=['days_last_login', 'days_clientSince'])
features.loc[:, ['days_last_login', 'days_clientSince']] = scaled_data
# features.head()
# features.head()
# 3.2 one hot encoding the status and gender columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cols = encoder.fit_transform(features[['gender', 'status']])
encoded_df = pd.DataFrame(
    encoded_cols, columns=encoder.get_feature_names_out(['gender', 'status']))
features_encoded = pd.concat(
    [features.drop(['gender', 'status'], axis=1), encoded_df], axis=1)
# features_encoded.describe()
# features_encoded.columns
# 3.4 testing clustering methods
# kmeans
k = 5
kmeans = KMeans(n_clusters=k, random_state=100)
kmeans_clusters = kmeans.fit_predict(features_encoded)
# pd.Series(kmeans_clusters).value_counts()

# hierarichal clustering
h = 5
hierarchical = AgglomerativeClustering(n_clusters=h)
hi_clusters = hierarchical.fit_predict(features_encoded)


# 3.5 measuring the performance of clustering methods
kmeans_sil = silhouette_score(features_encoded,
                              kmeans_clusters)
hi_sil = silhouette_score(features_encoded,
                          hi_clusters)
# print(
#     f"Hierarchical clustering silhouette score: {hi_sil}, for the kmeans {kmeans_sil} ")

clusters = pd.Series(kmeans_clusters, name='clusters') + 1
# features.loc[:, 'clusters'] = clusters

output = pd.concat([id, clusters], axis=1)
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
output['last_update'] = current_datetime

# Load
# features.to_csv("clusters.csv", index=False)

output.to_sql('clusters', con=engine, if_exists='replace', index=False)




# 3.6 visualizing the results

# cluster distribution 
sns.countplot(x='clusters', data=features)
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()


# gender distribution
plt.figure(figsize=(12, 10))
sns.countplot(x='clusters', hue='gender', data=features)
plt.title('Gender Distribution in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()

# isActive distributon
sns.countplot(x='clusters', hue='isActive', data=features)
plt.title('Active Status Distribution in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Is Active', labels=['Inactive', 'Active'])
plt.show()


sns.boxplot(x='clusters', y='days_last_login', data=features)
plt.title('Days Since Last Login by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Days Since Last Login')
plt.show()

sns.boxplot(x='clusters', y='days_clientSince', data=features)
plt.title('Client since by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Client since')
plt.show()
