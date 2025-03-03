from datetime import datetime

import pymysql
import pandas as pd

# 📂 CSV file paths
predictions_csv = '/home/ioana/Desktop/Model_Licenta/output/final_predictions_with_time.csv'
best_predictions_csv = '/home/ioana/Desktop/Model_Licenta/output/best_model_predictions_with_time.csv'
losses_csv = '/home/ioana/Desktop/Model_Licenta/output/training_results.csv'

test_set_predictions="/home/ioana/Desktop/Model_Licenta/output/test_set_predictions.csv"
test_results= "/home/ioana/Desktop/Model_Licenta/output/test_results.csv"

# 📋 Read CSV data
try:
    predictions_data = pd.read_csv(best_predictions_csv)
    print(f"Predictions CSV loaded successfully with {len(predictions_data)} records.")

    losses_data = pd.read_csv(losses_csv)
    print(f" Losses CSV loaded successfully with {len(losses_data)} records.")
    losses_data["Time"] = pd.to_datetime(losses_data["Time"])

    # TESTT
    predictions_data_test = pd.read_csv(test_set_predictions)
    print(f"TEST Predictions CSV loaded successfully with {len(predictions_data_test)} records.")

    losses_data_test = pd.read_csv(test_results)
    print(f" TEST Losses CSV loaded successfully with {len(losses_data_test)} records.")
    losses_data_test["Time"] = pd.to_datetime(losses_data_test["Time"])


except FileNotFoundError as e:
    print(f" Error: {e}")
    exit(1)

#  Database connection details
db_config = {
    'host': '172.19.0.4',
    'user': 'root',
    'password': 'pass',
    'database': 'lic_feb3'
}

try:
    #  Connect to MySQL
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()

    # Insert Predictions into prediction_hr
    insert_prediction_query = """
    INSERT INTO prediction_t1 (user_id, time, real_value, predicted_value)
    VALUES (%s, %s, %s, %s);
    """

    for _, row in predictions_data.iterrows():
        cursor.execute(insert_prediction_query, (
            int(row['user_id']),
            row['time'],
            float(row['real_value']),
            float(row['predicted_value'])
        ))

    prediction_rows = cursor.rowcount
    print(f" {prediction_rows} rows inserted into 'prediction_t1' table.")

    #  Insert Losses into first_loss, all metrics

    insert_loss_query = """
    INSERT INTO loss_t1 (epoch,time, train_loss, val_loss,MAE,MSE,R2)
    VALUES (%s, %s, %s,%s, %s, %s, %s);
    """

    for _, row in losses_data.iterrows():
        cursor.execute(insert_loss_query, (
            int(row['Epoch']),
            row["Time"].strftime('%Y-%m-%d %H:%M:%S'),
            float(row['Train Loss']),
            float(row['Validation Loss']),
            float(row['MAE']),
            float(row['MSE']),
            float(row['R2'])
        ))

    loss_rows = cursor.rowcount
    print(f" {loss_rows} rows inserted into 'loss' table.")

    #TESTTTT
    # Insert Predictions into prediction_hr
    insert_prediction_test_query = """
        INSERT INTO test_t1 (user_id, time, real_value, predicted_value)
        VALUES (%s, %s, %s, %s);
        """

    for _, row in predictions_data_test.iterrows():
        cursor.execute(insert_prediction_test_query, (
            int(row['user_id']),
            row['time'],
            float(row['real_value']),
            float(row['predicted_value'])
        ))

    prediction_rows_test = cursor.rowcount
    print(f"TESTT {prediction_rows_test} rows inserted into 'prediction_hr' table.")

    #  Insert Losses into first_loss, all metrics

    insert_loss_query_test = """
        INSERT INTO test_loss_t1 (time,MAE,MSE,R2)
        VALUES (%s, %s, %s,%s);
        """

    for _, row in losses_data_test.iterrows():
        cursor.execute(insert_loss_query_test, (
            row["Time"].strftime('%Y-%m-%d %H:%M:%S'),
            float(row['MAE']),
            float(row['MSE']),
            float(row['R2'])
        ))

    loss_rows_test = cursor.rowcount
    print(f"TEST  {loss_rows_test} rows inserted into 'loss' table.")

    #  Commit all changes
    connection.commit()

except pymysql.MySQLError as e:
    print(f" MySQL Error: {e}")
finally:
    if connection:
        connection.close()
        print("Database connection closed.")
