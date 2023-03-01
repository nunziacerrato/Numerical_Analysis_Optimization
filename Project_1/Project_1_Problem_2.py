import numpy as np
import scipy.linalg
import logging
import qutip


def wilkin(n):
    W = np.tril(-np.ones((n,n)),k=-1) + np.eye(n)
    W[:,n-1] = 1
    return W


if __name__=='__main__':

    logging.basicConfig(level=logging.WARNING)
    n = 60
    for n in range(2,61):
        W = wilkin(n)
        logging.debug(f'W = {W}')

        vect = np.ones(n)
        b = W @ vect
        x = np.linalg.solve(W,b)
        #x = scipy.linalg.inv(W) @ b

        error = sum(np.abs(x-vect))
        treshold = 1e-15
        if error <= treshold:
            logging.info(f'n = {n}')
            logging.info(f'||x - e||_1 = {error}')
        elif error > treshold:
            logging.warning(f'n = {n}')
            print(x)
            logging.warning(f'||x - e||_1 = {error}')
            for i in range(n):
                x_i = x[i]
                if abs(x_i-1) > treshold:
                    logging.warning(f'x[{i}] = {x[i]}')



