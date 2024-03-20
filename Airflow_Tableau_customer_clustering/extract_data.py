from sqlalchemy import create_engine, text
import pandas as pd

def read_credentials(filename):
    credentials = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split(' = ')
            credentials[key] = value.strip("'")
    return credentials

def extract_data():
    credentials = read_credentials('/Users/sajad/Airflow/scripts/cred.text')
    engine = create_engine(
        f"mysql+mysqlconnector://{credentials['user_name']}:{credentials['password_enc']}@{credentials['host_name']}/{credentials['db_name']}")

    query = text("""select id, gender,
                        isActive, `status`, last_login, clientSince
                    from users_db.mock_data""")
    df = pd.read_sql(query, engine)
    print(f"mock_data table extracted successfully.\nnumber of rertreived rows and columns: {df.shape}\n{df.head(2)}")
    return df.to_json()
