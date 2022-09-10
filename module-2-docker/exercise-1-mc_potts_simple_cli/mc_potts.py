import numpy as np
from numpy.random import default_rng


def mc_potts(n=50, t=25, kT=0.9, seed=None):
    """
    Simple Monte Carlo Potts model with uniform boundary energy and mobility.
    
    Parameters
    ----------
    n: int
        length of square system, number of pixels
    t: int
        number of time steps to run simulation for
    kT: float
        Boltzmann constant * temperature
    seed: Optional[int]
        random seed to use for MC simulation
    
    Returns
    ----------
    x: ndarray
        n x n array of final states after running the simulation
    """
    n2 = n**2 # number of sites in system
    
    # for selecting neighborhood around site of interest
    col_offsets = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    row_offsets = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    
    x = np.reshape(np.arange(n2), (n, n)) # initialize each pixel as its own nucleus
    rng = default_rng(seed)  # seed the random number generator
    for _ in range(t): # MC step
        for _ in range(n2): # iteration within MC step
            r, c  = rng.integers(n, size=2)  # select random row and column indices
            # get neighborhood and apply periodic boundary conditions
            neighbors = x[(r+row_offsets) % n, (c  + col_offsets) % n]
            # choose a new state to transition to from the unique set of neighbors
            new_state = rng.choice(np.unique(neighbors))
            # compute energy change of transition
            dE = (neighbors == x[r,c]).sum()-(neighbors == new_state).sum()
            # accept or reject transition with metropolis probability
            if dE <= 0:
                x[r,c] = new_state
            elif rng.random() < np.exp(-dE/kT):
                x[r,c] = new_state
            
    return x