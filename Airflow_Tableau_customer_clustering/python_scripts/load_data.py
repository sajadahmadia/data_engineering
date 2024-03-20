from sqlalchemy import create_engine
import pandas as pd

def read_credentials(filename):
    credentials = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split(' = ')
            credentials[key] = value.strip("'")
    return credentials



def load_data(**kwargs):
    ti = kwargs['ti']
    df_json = ti.xcom_pull(task_ids='transform_data')
    df = pd.read_json(df_json)
    credentials = read_credentials('/Users/sajad/Airflow/scripts/cred.text')
    engine = create_engine(
        f"mysql+mysqlconnector://{credentials['user_name']}:{credentials['password_enc']}@{credentials['host_name']}/{credentials['db_name']}")

    df.to_sql('airflow_data', engine, if_exists='replace', index=False)
