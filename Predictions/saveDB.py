import pymysql
import pandas as pd

# 📂 CSV file paths
predictions_csv = '/home/ioana/Desktop/Model_Licenta/output2/predictions_transfer_learning.csv'
losses_csv = '/home/ioana/Desktop/Model_Licenta/output2/train_val_losses.csv'

# 📋 Read CSV data
try:
    predictions_data = pd.read_csv(predictions_csv)
    print(f"Predictions CSV loaded successfully with {len(predictions_data)} records.")

    losses_data = pd.read_csv(losses_csv)
    print(f" Losses CSV loaded successfully with {len(losses_data)} records.")

except FileNotFoundError as e:
    print(f" Error: {e}")
    exit(1)

#  Database connection details
db_config = {
    'host': '172.19.0.3',
    'user': 'root',
    'password': 'pass',
    'database': 'lic_feb1'
}

try:
    #  Connect to MySQL
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()

    # Insert Predictions into prediction_hr
    insert_prediction_query = """
    INSERT INTO prediction_hr (user_id, time, real_value, predicted_value)
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
    print(f" {prediction_rows} rows inserted into 'prediction_hr' table.")

    #  Insert Losses into loss
    insert_loss_query = """
    INSERT INTO loss (epoch, train_loss, val_loss)
    VALUES (%s, %s, %s);
    """

    for _, row in losses_data.iterrows():
        cursor.execute(insert_loss_query, (
            int(row['epoch']),
            float(row['train_loss']),
            float(row['val_loss'])
        ))

    loss_rows = cursor.rowcount
    print(f" {loss_rows} rows inserted into 'loss' table.")

    #  Commit all changes
    connection.commit()

except pymysql.MySQLError as e:
    print(f" MySQL Error: {e}")
finally:
    if connection:
        connection.close()
        print("🔒 Database connection closed.")
