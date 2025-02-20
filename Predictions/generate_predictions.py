import pickle
import argparse
import torch
import yaml
import numpy as np
from datetime import datetime
import pandas as pd
import os

from Predictions.path_config import MODEL2_PMDATA_PATH, BASE_PATH
from jan_transfer_one.sec_model import TransformerModel

parser = argparse.ArgumentParser(description='')
parser.add_argument('--device', default='cpu')
parser.add_argument('--modelfolder', type=str, default='')
parser.add_argument('--nsample', type=int, default=5)
args = parser.parse_args()

config = yaml.safe_load(open(BASE_PATH, 'r'))
# median, iqr = pickle.load(open(MEAN_PATH, 'rb'))

def get_model():
    model = TransformerModel(config, args.device).to(args.device)
    model.load_state_dict(torch.load(MODEL2_PMDATA_PATH))
    model.eval()
    return model


def convert_tensor_to_datetime(tensor_data):
    if tensor_data.is_cuda:
        tensor_data = tensor_data.cpu()

    numpy_array = tensor_data.numpy()
    date_time_list = np.vectorize(datetime.fromtimestamp)(numpy_array).astype(str)

    return date_time_list


import pandas as pd
import os

import pandas as pd
import os

def save_predictions_to_csv(y_true, y_pred, times, user_ids, filename='predictions_with_time.csv'):
    """
    Save real and predicted values to a CSV file along with timestamps after denormalization.

    Args:
    - y_true: List of real denormalized values.
    - y_pred: List of predicted denormalized values.
    - times: List of timestamps corresponding to predictions.
    - user_ids: List of user IDs corresponding to each prediction.
    - filename: The output CSV filename.
    """
    predictions_df = pd.DataFrame({
        "user_id": user_ids,
        "time": times,
        "real_value": y_true,
        "predicted_value": y_pred
    })

    output_dir = '/home/ioana/Desktop/Model_Licenta/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, filename)
    predictions_df.to_csv(file_path, index=False)
    print(f"Predictions saved to: {file_path}")


def process_batch_all_batch(batch_file, batch_index):
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f)

    model = get_model()
    print(f"Processing batch file: {batch_file}")
    print(f"Samples X shape: {batch['samples_x'].shape}")
    print(f"Samples Y shape: {batch['samples_y'].shape}")
    print(f"Info shape: {batch['info'].shape}")
    print(f"Time shape: {batch['time'].shape}")
    print(f"Activity shape: {batch['activity'].shape}")
    print(f"Pause shape: {batch['pause'].shape}")

    samples_x = batch['samples_x'].to(model.device).float()
    samples_y = batch['samples_y'].to(model.device).float()
    info = batch['info'].to(model.device)
    time = batch['time'].to(model.device)
    activity = batch['activity'].to(model.device)
    pause = batch['pause'].to(model.device)

    info_np = batch['info'][batch_index].unsqueeze(0).float()
    datetime_list = convert_tensor_to_datetime(time)

    denormalized_real = (samples_y[:, 2] * iqr) + median

    with torch.no_grad():
        forecasted_values = model.forecast(samples_x, samples_y, activity, n_samples=5)

    denormalized_forecast = (forecasted_values * iqr) + median

    predictions = []
    lx = int(info_np[0][1])

    for t in range(denormalized_forecast.shape[1]):
        predictions.append({
            "activity": find_activity(datetime_list[0][t]),
            "time": datetime_list[0][t],
            "predicted_value": denormalized_forecast[0, t].median().item(),
            "real_value": denormalized_real[0, t].item()
        })

    filename = f'predictions_{os.path.basename(batch_file).replace(".pkl", "")}_index_{batch_index}.csv'
    save_predictions_to_csv(predictions, filename)

