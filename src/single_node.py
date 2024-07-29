import logging
import sys
from pathlib import Path
import pandas as pd
import joblib
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tsfresh.feature_extraction import settings
from sklearn.model_selection import train_test_split
import time

from ndata_pipeline import DataPipeline
from vae import VAE

def process_node(node_dir, output_dir, repeat_num, expConfig_num):
    # Extract node name from directory
    node_name = os.path.basename(node_dir)
    train_path = os.path.join(node_dir, f'{node_name}_train.hdf')
    test_path = os.path.join(node_dir, f'{node_name}_test.hdf')

    # Load data using DataPipeline
    pipeline = DataPipeline()

    x_train, x_test = pipeline.load_HPC_data(train_path, test_path)

    if x_train is None or x_test is None:
        logging.error(f"Data loading failed for node {node_name}")
        return

    # Ensure index column is formatted correctly
    x_train['index'] = x_train['job_id'].astype(str) + '_' + x_train['component_id'].astype(str)
    x_test['index'] = x_test['job_id'].astype(str) + '_' + x_test['component_id'].astype(str)

    new_x_train = x_train.drop(['index', 'uid'], axis=1)
    new_x_test = x_test.drop(['index', 'uid'], axis=1)

    start_time = time.time()
    x_train_fe = pipeline.tsfresh_generate_features(new_x_train, fe_config="minimal")
    feature_extraction_time_train = time.time() - start_time

    start_time = time.time()
    x_test_fe = pipeline.tsfresh_generate_features(new_x_test, fe_config="minimal")
    feature_extraction_time_test = time.time() - start_time

    # Make the number of columns and the order equal
    if len(x_test_fe.columns) < len(x_train_fe.columns):
        x_train_fe = x_train_fe[x_test_fe.columns]
    elif len(x_test_fe.columns) > len(x_train_fe.columns):
        x_test_fe = x_test_fe[x_train_fe.columns]

    x_train_fe = x_train_fe[x_test_fe.columns]
    assert all(x_train_fe.columns == x_test_fe.columns)
    x_test_fe.reset_index(drop=True, inplace=True)

    x_train_scaled, x_test_scaled = pipeline.scale_data(x_train_fe, x_test_fe, save_dir=output_dir)

    input_dim = x_train_scaled.shape[1]
    intermediate_dim = int(input_dim / 2)
    latent_dim = int(input_dim / 3)

    vae = VAE(
        name="model",
        input_dim=input_dim,
        intermediate_dim=intermediate_dim,
        latent_dim=latent_dim,
        learning_rate=1e-4
    )

    # Train the VAE model
    start_time = time.time()
    vae.fit(
        x_train=x_train_scaled,
        epochs=1000,
        batch_size=32,
        validation_split=0.1,
        save_dir=output_dir,
        verbose=0
    )
    training_time = time.time() - start_time + feature_extraction_time_train

    deployment_metadata = {
        'threshold': vae.threshold,
        'raw_column_names': list(x_train_scaled.columns),
        'fe_column_names': settings.from_columns(list(x_train_scaled.columns)),
        'training_time': training_time
    }

    with open(Path(output_dir) / 'deployment_metadata.json', 'w') as fp:
        json.dump(deployment_metadata, fp)

    start_time = time.time()
    y_pred_test, x_test_recon_errors = vae.predict_anomaly(x_test_scaled)
    prediction_time = time.time() - start_time + feature_extraction_time_test

    result_dict = {
        "y_pred_test": np.array(y_pred_test).tolist(),
        "x_test_recon_errors": np.array(x_test_recon_errors).tolist(),
        "training_time": training_time,
        "prediction_time": prediction_time
    }

    result_file = Path(output_dir) / "results" / f"{node_name}.json"
    with open(result_file, "w") as outfile:
        json.dump(result_dict, outfile)

    logging.info(f"Results for {node_name} saved to {result_file}")

def main(repeat_nums, expConfig_nums, data_dir, pre_selected_features_filename, output_dir, verbose=False):
    
    logging.basicConfig(format='%(asctime)s %(levelname)-7s %(message)s', stream=sys.stderr, level=logging.INFO if verbose else logging.DEBUG)
        
    logging.info("This is an info message")
    logging.debug("This is a debug message")
    
    # Create output directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info("Created outputs directory")
    else:
        logging.info("Output directory already exists")

    if not os.path.exists(output_dir + "/results"):
        os.makedirs(output_dir + "/results")
        logging.info("Created results directory")
    else:
        logging.info("Results directory already exists")

    node_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]

    for node_dir in node_dirs:
        for repeat_num in repeat_nums:
            for expConfig_num in expConfig_nums:
                logging.info(f"Processing node {node_dir}, repeat_num {repeat_num}, expConfig_num {expConfig_num}")
                process_node(node_dir, output_dir, repeat_num, expConfig_num)
                logging.info(f"Completed processing node {node_dir}")

if __name__ == '__main__':
    repeat_nums = [0]
    expConfig_nums = [0]
    data_dir = "xue_code/prodigy_artifacts/ai4hpc_deployment/src/eclipse_small_prod_dataset"
    pre_selected_features_filename = None
    output_dir = "/THL5/home/shyunie/xue_code/prodigy_artifacts/prodigy_ae_output"
    verbose = True
    main(repeat_nums, expConfig_nums, data_dir, pre_selected_features_filename, output_dir, verbose)
    
    logging.info("Script is completed")
