import pandas as pd
import datetime
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
import numpy as np


def transform_data(**kwargs):
    try:
        ti = kwargs['ti']
        df_json = ti.xcom_pull(task_ids='extract_data')
        df = pd.read_json(df_json)
        user_id = df['id']
        df['last_login'] = pd.to_datetime(df['last_login'])
        df['clientSince'] = pd.to_datetime(df['clientSince'])
        df['isActive'] = df['isActive'].astype(int)
        features = df.drop(columns=['id'])
        features['days_last_login'] = (
            datetime.datetime.today() - features['last_login']).dt.days
        features = features.drop(columns=['last_login'])

        features['days_clientSince'] = (
            datetime.datetime.today() - features['clientSince']).dt.days
        features = features.drop(columns=['clientSince'])

        imputer = KNNImputer(n_neighbors=5, weights="uniform")
        imputed_data = imputer.fit_transform(features[['days_last_login']])
        features.loc[:, 'days_last_login'] = imputed_data

        # status and gender with mode
        imputer = SimpleImputer(strategy='most_frequent', missing_values=None)
        imputed_data = imputer.fit_transform(features[['gender', 'status']])
        imputed_df = pd.DataFrame(imputed_data, columns=['gender', 'status'])
        features[['gender', 'status']] = imputed_df

        scaler = MinMaxScaler(feature_range=(0, 1))
        days_data = features[['days_last_login', 'days_clientSince']]
        scaled_data = scaler.fit_transform(days_data)
        features = features.drop(
            columns=['days_last_login', 'days_clientSince'])
        features.loc[:, ['days_last_login', 'days_clientSince']] = scaled_data

        encoder = OneHotEncoder(sparse =False, drop='first')
        encoded_cols = encoder.fit_transform(features[['gender', 'status']])
        encoded_df = pd.DataFrame(
            encoded_cols, columns=encoder.get_feature_names_out(['gender', 'status']))
        features_encoded = pd.concat(
            [features.drop(['gender', 'status'], axis=1), encoded_df], axis=1)
        print("\n",features_encoded.isna().sum())
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=100)
        kmeans_clusters = kmeans.fit_predict(features_encoded)
        clusters = pd.Series(kmeans_clusters, name='clusters') + 1
        print(f"\nclusters created successfully\nsample:\n{clusters.head()}")
        output = pd.concat([user_id, clusters], axis=1)
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        output['last_update'] = current_datetime
        return output.to_json()
    except Exception as err:
        raise ValueError(f"error in transforming the data: {err} ")
    
    # Data transformation steps...
    # Example transformation:
    # ti = kwargs['ti']
    # df_json = ti.xcom_pull(task_ids='extract_data')
    # df = pd.read_json(df_json)
    # df['last_login'] = pd.to_datetime(df['last_login'])
    # df['days_last_login'] = (datetime.now() - df['last_login']).dt.days

    # # Continue with the rest of your transformation steps...
    # # Assuming you conclude with clustering and adding 'clusters' and 'last_update' columns
    # imputer = SimpleImputer(strategy='mean')
    # df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])))
    # df_imputed.columns = df.select_dtypes(include=[np.number]).columns
    # df[df_imputed.columns] = df_imputed
    # kmeans = KMeans(n_clusters=5, random_state=100)
    # df['clusters'] = kmeans.fit_predict(df[['days_last_login']]) + 1
    # df['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # return df.to_json()
