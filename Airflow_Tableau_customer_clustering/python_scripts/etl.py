#!/Users/sajad/Documents/legionella/legionella_py/bin/python
from sqlalchemy import create_engine, text
import pandas as pd
import sys
import datetime
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans


def read_credentials(filename):
    """
    Args:
        filename (str): The path to the file containing the database credentials.

    Returns:
        dict: A dictionary containing the database credentials, 
        with keys like 'host_name', 'db_name', 'user_name', and 'password_enc'.
    """
    credentials = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split(' = ')
            credentials[key] = value.strip("'")
    return credentials


# reading the credentials
def extract(credentials_file):
    """
    Extracts data from a MySQL database using provided credentials.
    Args:
    credentials_file (str): The path to the file containing the database credentials.

    Returns:
    DataFrame: A pandas DataFrame containing the extracted data from the 'mock_data' table.
    """
    try:
        credentials = read_credentials(credentials_file)
        host_name = credentials['host_name']
        db_name = credentials['db_name']
        user_name = credentials['user_name']
        password_enc = credentials['password_enc']

        # establishing the connection, extracating the required columns
        engine = create_engine(
            f'mysql+mysqlconnector://{user_name}:{password_enc}@{host_name}/{db_name}')

        query = text("""select id, gender,
                        isActive, `status`, last_login, clientSince
                    from users_db.mock_data""")

        df = pd.read_sql(query, engine)
        print(
            f"mock_data table extracted successfully.\nnumber of rertreived rows and columns: {df.shape}\n{df.head(2)}")
        return df
    except Exception as err:
        raise ValueError(f"error in connection to the database: {err}")


def transform(df):
    """
    Transforms the extracted DataFrame by performing data cleaning
    and feature engineering steps, including handling missing values,
    encoding categorical variables, and clustering.

    Args:
    df (DataFrame): The pandas DataFrame containing the extracted data.

    Returns:
    tuple: A tuple containing the user IDs (as a pandas Series)
    and the cluster assignments (as a pandas Series) for each row in the DataFrame.
    """
    try:
        user_id = df['id']
        df['last_login'] = pd.to_datetime(df['last_login'])
        df['clientSince'] = pd.to_datetime(df['clientSince'])
        df['isActive'] = df['isActive'].map({'0': 0, '1': 1})
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

        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_cols = encoder.fit_transform(features[['gender', 'status']])
        encoded_df = pd.DataFrame(
            encoded_cols, columns=encoder.get_feature_names_out(['gender', 'status']))
        features_encoded = pd.concat(
            [features.drop(['gender', 'status'], axis=1), encoded_df], axis=1)

        k = 5
        kmeans = KMeans(n_clusters=k, random_state=100)
        kmeans_clusters = kmeans.fit_predict(features_encoded)
        clusters = pd.Series(kmeans_clusters, name='clusters') + 1
        print(f"\nclusters created successfully\nsample:\n{clusters.head()}")
        return user_id, clusters
    except Exception as err:
        raise ValueError(f"error in transforming the data: {err} ")


def load(user_id, clusters, credentials_file):
    """
    Loads the transformed data into a MySQL database, creating a new table named 'clusters'.

    Args:
    user_id (Series): A pandas Series containing user IDs.
    clusters (Series): A pandas Series containing cluster assignments for each user.
    credentials_file (str): The path to the file containing the database credentials.

    Returns:
    None
    """
    try:
        credentials = read_credentials(credentials_file)
        engine = create_engine(
            f'mysql+mysqlconnector://{credentials["user_name"]}:{credentials["password_enc"]}@{credentials["host_name"]}/{credentials["db_name"]}')

        output = pd.concat([user_id, clusters], axis=1)
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        output['last_update'] = current_datetime
        output.to_sql('clusters', con=engine, if_exists='replace', index=False)
        print(
            f"\nData loaded successfully into clusters table.final result\n{output.head()}")
    except Exception as err:
        print(f"Failed to load data into database: {err}")


def main():
    if len(sys.argv) != 2:
        print("Usage: script.py <path_to_credentials_file>")
        sys.exit(1)

    credentials_file = sys.argv[1]
    df = extract(credentials_file)
    user_id, clusters = transform(df)
    load_result = load(user_id, clusters, credentials_file)


if __name__ == '__main__':
    main()
