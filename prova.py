
import numpy as np 


A = np.array([[1,2,3],[4,5,6],[7,8,9]])


def mask_nd(x, m):
    '''
    Mask a 2D array and preserve the
    dimension on the resulting array
    ----------
    x: np.array
       2D array on which to apply a mask
    m: np.array
        2D boolean mask  
    Returns
    -------
    List of arrays. Each array contains the
    elements from the rows in x once masked.
    If no elements in a row are selected the 
    corresponding array will be empty
    '''
    take = m.sum(axis=1)
    return np.split(x[m], np.cumsum(take)[:-1])

def upper_tri_masking(A):
    m = A.shape[0]
    print(f'm = {m}')
    r = np.arange(m)
    print(f'r = {r}')
    mask = r[:,None] < r
    print(r[:,None])
    print(f'mask = {mask}')
    return mask_nd(A, mask)
    # return A[mask]

B = upper_tri_masking(A)

print(B)