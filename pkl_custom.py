import pickle as pkl
import numpy as np

def open_pkl_file(filename:str):
    with open(filename, 'rb') as file:
        extract = pkl.load(file)
        
    file.close()
    
    return extract

def save_pkl_file(filename:str, object):
    with open(filename, 'wb') as file:
        pkl.dump(object, file)
        
    file.close()
    
def load_graph_info(input):
    if isinstance(input, np.ndarray):
        return input
    elif isinstance(input, str):
        return open_pkl_file(input)
    else:
        raise ValueError("input must be either a Numpy array or a string of the file path of a pickle file")