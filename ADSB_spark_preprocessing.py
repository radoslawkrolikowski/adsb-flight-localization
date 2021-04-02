import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


def preprocess_adsb(row, max_measurements, n_meas_diff):
    """Performs the ads-b data preprocessing that includes:
    - exploding the measurements JSON array,
    - extracting the sensor, timestamp and RSSI information from an array of measurements,
    - conducting timestamps synchronization,
    - adding the sensor localization data,
    - performing the feature extraction,
    - filling missing values.

    Parameters
    ----------
    row: pandas.Series
        Row of raw ads-b data
    max_measurements: int    
        Maximum number of measurements to take into consideration
    n_meas_diff: int
        Number of measurements to be taken into account while calculating differences

    Returns
    -------
    df: pandas.DataFrame
        Dataframe of preprocessed ads-b data

    """

    # Convert row to pandas dataframe
    df = row.to_frame().transpose()

    # Explode measurements data
    meas = df.measurements.apply(lambda row: json.loads(row)[:max_measurements])
    meas = meas.apply(pd.Series)
    
    df.drop(columns='measurements', inplace=True)
    
    df = pd.concat([df, meas], axis=1)

    meas_list = []    
    
    for col in range(max_measurements):

        try:
            columns = ['sensor_{}'.format(col), 'tmp_{}'.format(col), 'RSSI_{}'.format(col)]

            meas_list.append(pd.DataFrame(df[col].to_list(), columns=columns))

            df.drop(columns=col, inplace=True)

        except KeyError:
            empty_cols = {'sensor_{}'.format(col) : np.NaN, 'tmp_{}'.format(col) : 0,
                          'RSSI_{}'.format(col) : 0}
            
            meas_list.append(pd.DataFrame(empty_cols, index=[0]))
            
    # Concatenate all columns
    meas = pd.concat(meas_list, axis=1)
    meas.reset_index(inplace=True)
    df.reset_index(inplace=True)
    df = pd.concat([df, meas], axis=1)
    
    # df.astype({'tmp_{}'.format(i): int for i in range(max_measurements)})
           
    # Replace tmp_0 with its absolute value
    df['tmp_0'] = df['tmp_0'].abs().astype('int')
    
    # Read sensors data
    sensors = pd.read_csv('round2_training/round2/round2_sensors.csv')
    
    # Read the synchronization coefficients
    with open(r"coeff_dict.pickle", "rb") as output_file:
        coeff_dict = pickle.load(output_file)

    sensors.set_index('serial', inplace=True)
    
    # Add the sensor localization data
    for i in range(max_measurements):
         df = df.join(sensors.loc[:, ['latitude', 'longitude', 'height']], on='sensor_{}'.format(i),
                               rsuffix='_{}'.format(i))
        
         df.rename({'height': 'height_{}'.format(i)}, axis=1, inplace=True)

    # Perform timestamp synchronization
    corrected_tmp = {}
    
    s1 = df.sensor_0
    timeAtServer = df.timeAtServer
        
    for i in range(1, max_measurements):
        
        corrected_tmp.setdefault(i, [])
        
        s2 = getattr(df, 'sensor_{}'.format(i))
        tmp_2 = getattr(df, 'tmp_{}'.format(i))

        if tmp_2[0] == 0:
            corrected_tmp[i].append(0)
            continue
        
        # Return the correction coefficients (default=(0,0))
        m, b = coeff_dict.get('{}_{}'.format(int(s1), int(s2)), (0, 0))
        
        corr = m * timeAtServer + b
        
        corrected_tmp[i].append(int(tmp_2 - corr))
        
    for i in range(1, max_measurements):
        df['tmp_{}'.format(i)] = pd.Series(corrected_tmp[i])
        
    df.drop(columns=['id', 'numMeasurements'] + ['sensor_{}'.format(i) for i in range(max_measurements)], inplace=True)
    
    # Feature extraction
    # Calculate the timestamp differences
    for col_1 in range(1, n_meas_diff):
        for col_2 in range(col_1):
            df['diff_{}_{}'.format(col_1, col_2)] =  int(df['tmp_{}'.format(col_1)]) - int(df['tmp_{}'.format(col_2)])
    
    
    # Calculate the mean latitude
    df['mean_lat'] = df[[col for col in df.columns if str(col).startswith('latitude_')]].mean(axis=1)
    
    # Calculate the mean longitude
    df['mean_lon'] = df[[col for col in df.columns if str(col).startswith('longitude_')]].mean(axis=1)

    # Mask NaN values
    mask = ~np.isnan(df[[col for col in df.columns if str(col).startswith('latitude_')]])

    # Calculate the weighted average of latitude
    df['w_mean_lat'] = float(np.average(df[[col for col in df.columns if str(col).startswith('latitude_')]].to_numpy()[mask],
                                  axis=0,
                                  weights=1/abs(df[[col for col in df.columns if str(col).startswith('tmp_')]].to_numpy()[mask])))
    
    # Calculate the weighted average of longitude
    df['w_mean_lon'] = float(np.average(df[[col for col in df.columns if str(col).startswith('longitude_')]].to_numpy()[mask],
                                  axis=0,
                                  weights=1/abs(df[[col for col in df.columns if str(col).startswith('tmp_')]].to_numpy()[mask])))
       
    # Fill missing values
    values = {}
    
    for col in df.columns:
        if ('latitude' in col) or ('mean_lat' in col):
            values[col] = -90
        elif ('longitude' in col) or ('mean_lon' in col):
            values[col] = -180
        else:
            values[col] = 0
                   
    df.drop(columns=['index'], inplace=True)
    df.fillna(value=values, inplace=True)
    
    return df
    