import numpy as np
from sklearn.model_selection import train_test_split

def get_batches(x, y, batch_size=50, whole_batch=True):
    n_batches = len(x)//batch_size
    if not whole_batch:
        n_batches += 1
    
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]

    batches = []
    for idx in range(0, len(x), batch_size):
        batches += [(x[idx:idx+batch_size], y[idx:idx+batch_size])]
    return batches

def get_sequence_batches(self, int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    # Get the number of characters per batch and number of batches we can make
    characters_per_batch = batch_size * seq_length
    n_batches = int(len(int_text)/characters_per_batch)
    
    # Keep only enough characters to make full batches
    arr = np.array(int_text[:n_batches*characters_per_batch])
    
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    batches = np.array([])
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:,n:n+seq_length]
        # The targets, shifted by one
        y_temp = arr[:,n+1:n+seq_length+1]
        
        y = np.zeros(x.shape, dtype=x.dtype)
        y[:,:y_temp.shape[1]] = y_temp
        
        if y.shape[1] != y_temp.shape[1]:
            for l in range(y.shape[0]):
                y[l][-1] = arr[(l+1)%len(arr)][0]
        
        new_batch = np.array([[x,y]])
        
        if batches.shape == (0,):
            batches = new_batch
        elif new_batch[0].shape == batches[0].shape:
            batches = np.append(batches, new_batch, axis=0)
            
    return np.array(batches)

def get_train_validation(x, y, valid_size=None, random_state=None):
    return train_test_split(x, y, test_size=valid_size, random_state=random_state)
