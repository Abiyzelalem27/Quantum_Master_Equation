

import numpy as np


def decaying_two_level_atom(times, gamma):
    """Return rho(t) for an atom initially in the excited state.

    Basis order: |g>, |e>.
    gamma: decay rate, in inverse units of time.
    """
    times = np.asarray(times, dtype=float)

    if np.any(times < 0):
        raise ValueError("Times must be non-negative.")
    if gamma < 0:
        raise ValueError("Decay rate gamma must be non-negative.")

    excited = np.exp(-gamma * times)
    rho = np.zeros(times.shape + (2, 2), dtype=complex)
    rho[..., 0, 0] = 1 - excited  # ground-state population
    rho[..., 1, 1] = excited      # excited-state population
    return rho
