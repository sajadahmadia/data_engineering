import sys
sys.path.append('the_path_to_your_airflow_home_directory')
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
#importing the external python scripts form the scripts directory
from scripts.extract_data import extract_data
from scripts.transform_data import transform_data
from scripts.load_data import load_data


default_args = {
    'owner' : 'your_user_name_for_airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=30),    
}

with DAG(dag_id='user_clustering',
    description = 'clustering users based on kmeans algorithm',
    default_args = default_args,
    schedule_interval = timedelta(days=1)
) as dag:
    
    extract_data_task = PythonOperator(
        task_id = 'extract_data',
        python_callable= extract_data
    )

    transform_data_task = PythonOperator(
        task_id = 'transform_data',
        python_callable= transform_data,
        provide_context=True
    )
    
    load_data_task = PythonOperator(
        task_id = 'load_data',
        python_callable= load_data,
        provide_context=True
    )
# determining the dependencies between tasks 
extract_data_task >> transform_data_task >> load_data_task
