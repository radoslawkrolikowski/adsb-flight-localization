import torch
from torch.utils.data.dataset import Dataset
import h5py
import numpy as np


class DatasetFromHDF5(Dataset):
    """Implementation of the Pytorch HDF5 Dataset.
    
    Parameters
    ----------
    filename: str
        Path to the hdf5 file
    dataset_x: str
        Name of the hdf5 dataframe containing input features
    dataset_y: str
        Name of the hdf5 dataframe containing target variables
    norm_params: dict
        Dictionary of normalization parameters
    feat_cols: list, optional (default=None)
        List of input column names. Use all columns if None
    load_all: boolean, optional (default=False)
        Whether to load all data from HDF5 at once. It will speed up processing, but
        the data will also take up a lot of memory.
    
    Returns
    -------
    x_norm: torch.DoubleTensor
        Tensor of normalized input features
    y_norm: torch.DoubleTensor
        Tensor of normalized target variables
        
    """
    
    def __init__(self, filename, dataset_x, dataset_y, norm_params, feat_cols=None, load_all=False):
                
        h5f = h5py.File(filename, 'r')
        
        if not feat_cols:
            self.feat_cols = norm_params['input_features']
        else:
            self.feat_cols = feat_cols
            
        self.n_features = len(self.feat_cols)
        
        if not load_all:
            self.x = h5f[dataset_x]['table']
            self.y = h5f[dataset_y]['table']
        else:
            self.x = h5f[dataset_x]['table'][:]
            self.y = h5f[dataset_y]['table'][:]           
        
        # Extract normalization parameters
        self.x_min = np.array([norm_params[col]['min'] for col in norm_params['input_features']])
        self.x_max = np.array([norm_params[col]['max'] for col in norm_params['input_features']])

        self.y_min = np.array([norm_params[col]['min'] for col in norm_params['target']])
        self.y_max = np.array([norm_params[col]['max'] for col in norm_params['target']])
               
        self.cols_idx = [idx for idx, col in enumerate(norm_params['input_features']) if col in self.feat_cols]


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        
        x_norm = (self.x[index][1] - self.x_min) / (self.x_max - self.x_min)
        y_norm = (self.y[index][1] - self.y_min) / (self.y_max - self.y_min)
        
        if self.feat_cols:
            x_norm = x_norm[self.cols_idx]
        
        return torch.DoubleTensor(x_norm), torch.DoubleTensor(y_norm)
    
    
class PandasDataset(Dataset):
    """Implementation of the Pytorch Dataset for Pandas dataframe.
    
    Parameters
    ---------
    df: pandas.DataFrame
        Pandas dataframe of input and target variables
    norm_params: dict
        Dictionary of normalization parameters
    feat_cols: list, optional (default=None)
        List of input column names. Use all columns if None
        
    Returns
    ------
    x_norm: torch.DoubleTensor
        Tensor of normalized input features
    y_norm: torch.DoubleTensor
        Tensor of normalized target variables  
    
    """
    
    def __init__(self, df, norm_params, feat_cols=None):
                
        if feat_cols:
            x_cols = feat_cols
        else:
            x_cols = norm_params['input_features']
            
        self.n_features = len(x_cols)
            
        y_cols = norm_params['target']
        
        # Extract normalization parameters
        x_min = np.array([norm_params[col]['min'] for col in x_cols])
        x_max = np.array([norm_params[col]['max'] for col in x_cols])

        y_min = np.array([norm_params[col]['min'] for col in y_cols])
        y_max = np.array([norm_params[col]['max'] for col in y_cols])
        
        self.x = (np.array(df[x_cols]) - x_min) / (x_max - x_min)
        self.y = (np.array(df[y_cols]) - y_min) / (y_max - y_min)
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return torch.DoubleTensor(self.x[idx]), torch.DoubleTensor(self.y[idx])
    

class MariaDB_chunk_indices:
    """Generates indices of database rows that form chunk of MySQL/MariaDB
    database parameterized by chunk_size (chunking is used to diminish memory usage while parsing).
    
    Parameters
    ---------
    cursor: mariadb.connection.cursor
        MariaDB/MySQL cursor object used to execute SQL statements
    table: str
        Name of the SQL table to read from
    chunk_size: int
        Size of the chunk of data to be read from database
        
    Returns
    -------
    chunk_indices[idx]: list
        List of chunk indices
        
    """

    def __init__(self, cursor, table, chunk_size):

        # Extract number of the rows in the database
        cursor.execute("SELECT COUNT(ID) FROM {};".format(table))
        db_length = cursor.fetchone()[0]
        
        # Calculate number of chunks
        num_chunks = db_length // chunk_size

        # Generate the indices list
        indices = np.arange(0, num_chunks * chunk_size)
        self.chunk_indices = np.array_split(indices, num_chunks)
        self.chunk_indices.append(np.arange(num_chunks * chunk_size, db_length))
        
    def __len__(self):
        
        return len(self.chunk_indices)
        
    def __getitem__(self, idx):
        
        return self.chunk_indices[idx]
    

class MariaDB_dataset(Dataset):
    """Implementation of the Pytorch dataset that reads rows specified by the indices parameter
    from the MariaDB/MySQL database. 
    
    Parameters
    ---------
    indices: np.array
        Array of indices
    cursor: mariadb.connection.cursor
        MariaDB/MySQL cursor object used to execute SQL statements
    table: str
        Name of the SQL table to read from
    norm_params: dict
        Dictionary of normalization parameters
    feat_cols: list, optional (default=None)
        List of input column names. Use all columns if None
        
    Returns
    ------
    x_norm: torch.DoubleTensor
        Tensor of normalized input features
    y_norm: torch.DoubleTensor
        Tensor of normalized target variables 
    
    """
    
    def __init__(self, indices, cursor, table, norm_params, feat_cols=None):

        super(MariaDB_dataset, self).__init__()
        
        indices = tuple(indices)
        
        if not feat_cols:
            feat_cols = norm_params['input_features']
        else:
            feat_cols = feat_cols
            
        y_cols = norm_params['target']
        
        self.n_features = len(feat_cols)
        
        # Extract normalization parameters
        x_min = np.array([norm_params[col]['min'] for col in feat_cols])
        x_max = np.array([norm_params[col]['max'] for col in feat_cols])

        y_min = np.array([norm_params[col]['min'] for col in y_cols])
        y_max = np.array([norm_params[col]['max'] for col in y_cols])
        
        feat_cols = ", ".join(feat_cols)
        y_cols = ", ".join(y_cols)
        
        # Fetch independent variables from database
        cursor.execute("SELECT {} FROM {} WHERE ID IN {};"\
               .format(feat_cols, table, indices))

        self.x = torch.DoubleTensor(cursor.fetchall())

        # Fetch target variables from database
        cursor.execute("SELECT {} FROM {} WHERE ID IN {};"\
               .format(y_cols, table, indices))

        self.y = torch.DoubleTensor(cursor.fetchall())
        
        # Perform normalization
        self.x = (self.x - x_min) / (x_max - x_min)
        self.y = (self.y - y_min) / (y_max - y_min)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

       