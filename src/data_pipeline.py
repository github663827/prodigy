import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters
from tsfresh.feature_extraction import settings

class DataPipeline():
    
    def __init__(self, **kwargs):
        """Initializes a `DataPipeline` object.
        
        Args:
            **kwargs: Dictionary containing the following optional keyword arguments:
                system_name (str): Name of the system (default is 'eclipse').
        """        
                
        self.window_size = 0
        self.dataset_name = kwargs.get('system_name', 'eclipse')                

        self.raw_features = None        
        self.fe_features = None
        
        self.logger = logging.getLogger(__name__)

    def load_HPC_data(self, train_path, test_path):
        """Loads data from the given file paths and returns the training and test data.
        
        Args:
            train_path (str): Path to the training data file.
            test_path (str): Path to the test data file.
            
        Returns:
            tuple: A tuple containing the training and test data.
        """        
        
        x_train = self._read_data(train_path)
        x_test = self._read_data(test_path)
        
        self.logger.info('Data read successfully')
        self.logger.info(f'Shape of x_train: {x_train.shape}')
        self.logger.info(f'Shape of x_test: {x_test.shape}')
                
        return x_train, x_test
        
    def generate_windows(self, data, window_size=60, skip_interval=15):
        """
        Generates rolling time windows for the input data, based on a given window size and skip interval.

        Args:
            data (pd.DataFrame): Input data to generate windows for.
            window_size (int): Size of the rolling window, in minutes. Defaults to 60.
            skip_interval (int): Number of minutes to skip between each window. Defaults to 15.

        Returns:
            pd.DataFrame: Dataframe containing the rolling time windows, with each row corresponding to a window.

        Raises:
            AssertionError: If the window size is 0.
        """
                
        assert window_size != 0, "Window size should be different than 0, to generate windows."
        
        data.reset_index(inplace=True)
        
        self.window_size = window_size
        
        data_windows = roll_time_series(
            data,
            column_id="component_id",
            column_sort="timestamp",
            max_timeshift=window_size,
            min_timeshift=window_size,
            rolling_direction=skip_interval,
        )   
            
        return data_windows
    
    def check_parameters(self, params):
        """
        Check if each parameter in params is allowed according to the allowed_values dictionary.

        Args:
            params (dict): Dictionary containing parameter names and their values.

        Raises:
            ValueError: If any parameter value is not in the allowed list.
        """
        
        allowed_values = {
                    'fe_config': ['minimal', 'efficient', None]
        }
        
        for param_name, param_value in params.items():
            if param_name not in allowed_values:
                continue
            if param_value not in allowed_values[param_name]:
                raise ValueError(f"Invalid value {param_value} for parameter {param_name}. Allowed values: {allowed_values[param_name]}")
    
    
    def tsfresh_generate_features(self, data, fe_config, kind_to_fc_parameters=None, column_id="uid", column_sort="timestamp"):
        """
        Extracts features from data using tsfresh library.

        Args:
            data (pd.DataFrame): Input data to extract features from.
            fe_config (str): Configuration of feature extractor. Can be "minimal" or "efficient".
            column_id (str): Name of column representing the ID of the time series.
            column_sort (str): Name of column representing the time of each observation.
            kind_to_fc_parameters (dict): Dictionary containing feature parameters for each feature kind.

        Raises:
            ValueError: If `fe_config` value is not in allowed list.

        Returns:
            pd.DataFrame: Extracted features.
        """        
        if data is None or len(data) == 0: 
            raise ValueError(f"Param [data] cannot be None or empty")
            
        self.check_parameters({'fe_config': fe_config})        
        
        if not (kind_to_fc_parameters is None):
            assert fe_config == None, "Either set fe_config or kind_to_fc_parameters, not both"
                                
        if np.any(pd.isnull(data)):
            self.logger.info(f'Raw time series: Before dropping NaNs: {data.shape}')
            data = data.dropna()
            self.logger.info(f'Raw time series:  Dropped NaNs: {data.shape}') 
        
        data['uid'] = data['job_id'].astype(str) + '_' + data['component_id'].astype(str)
        data.drop(columns=['job_id','component_id'],inplace=True)
        
        if kind_to_fc_parameters is None:
            self.logger.info("TSFRESH will use default_fc_parameters")
            data_fe = extract_features(            
                data,
                column_id=column_id,
                column_sort=column_sort,
                default_fc_parameters=EfficientFCParameters() if fe_config == 'efficient' else MinimalFCParameters(),
            )
        else:
            self.logger.info("TSFRESH will use kind_to_fc_parameters")
            data_fe = extract_features(            
                data,
                column_id=column_id,
                column_sort=column_sort,
                kind_to_fc_parameters=kind_to_fc_parameters,
            )                        
        data_fe.reset_index(inplace=True)
        data_fe[['job_id', 'component_id']] = data_fe['index'].str.split('_', expand=True)
        data_fe.drop(columns=['index'], inplace=True)
        
        if self.window_size == 0 :
            data_fe.set_index(["job_id", "component_id"],inplace=True)
        else:
            data_fe.set_index(["job_id", "component_id", "timestamp"],inplace=True)
        
        self.logger.info(f'Feature extraction: Before dropping NaNs: {data_fe.shape}')
        data_fe = data_fe.dropna(axis=1, how='any')    
        self.logger.info(f'Feature extraction: Dropped NaNs: {data_fe.shape}') 
                
        self.raw_features = list(data_fe.columns)
        self.fe_features = settings.from_columns(self.raw_features)
        
        return data_fe
    
    # def scale_data(self, x_train, x_test=None, save_dir=None):        
    #     """
    #     Scales data using MinMaxScaler.

    #     Args:
    #         x_train (pd.DataFrame): Training data to scale.
    #         x_test (pd.DataFrame, optional): Test data to scale. Defaults to None.
    #         save_dir (str, optional): Directory to save scaler object. Defaults to None.

    #     Returns:
    #         pd.DataFrame: Scaled training data.
    #         pd.DataFrame: Scaled test data.
    #     """        
    
    #     scaler = MinMaxScaler(feature_range=(0, 1), clip=True)

    #     x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)        
    #     if not (x_test is None):
    #         self.logger.info(f"x_test is not None, scaling")            
    #         x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
        
    #     if not (save_dir is None):
    #         scaler_filename = "scaler.save"
    #         joblib.dump(scaler, Path(save_dir) / scaler_filename)
    #         self.logger.info(f"Scaler is saved")            
            
    #     return x_train, x_test
    
    def scale_data(self, x_train, x_test=None, save_dir=None):        
        """
        Scales data using MinMaxScaler.

        Args:
            x_train (pd.DataFrame): Training data to scale.
            x_test (pd.DataFrame, optional): Test data to scale. Defaults to None.
            save_dir (str, optional): Directory to save scaler object. Defaults to None.

        Returns:
            pd.DataFrame: Scaled training data.
            pd.DataFrame: Scaled test data.
        """        
    
        # Ensure all columns are numeric
        x_train = x_train.apply(pd.to_numeric, errors='coerce')
        
        if x_test is not None:
            x_test = x_test.apply(pd.to_numeric, errors='coerce')

        scaler = MinMaxScaler(feature_range=(0, 1), clip=True)

        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)        
        
        if x_test is not None:
            self.logger.info(f"x_test is not None, scaling")
            x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
        
        if save_dir is not None:
            scaler_filename = "scaler.save"
            joblib.dump(scaler, Path(save_dir) / scaler_filename)
            self.logger.info(f"Scaler is saved")
        
        return x_train, x_test

    def _read_data(self, abs_input_path):
        """
        Reads data from HDF5 file.

        Args:
            abs_input_path (str): Absolute path to input data file.

        Returns:
            pd.DataFrame: Input data.
        """        
        
        try:
            data = pd.read_hdf(abs_input_path)
        except FileNotFoundError:
            self.logger.error(f"File not found!: {abs_input_path}")
            return None
        
        return data
