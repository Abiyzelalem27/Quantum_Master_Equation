

import numpy as np


def master_equation(hbar, rho, jump_operators, H):
    """
    Calculate the right-hand side of the Lindblad master equation:

        dρ/dt = -(i/hbar)[H, ρ] + sum_k (L_k ρ L_k† - 0.5 {L_k† L_k, ρ})

    Parameters
    ----------
    hbar : positive float
        Reduced Planck constant. Use 1 for natural units.
    rho : array_like, shape (d, d)
        Density matrix.
    jump_operators : iterable of arrays, each shape (d, d)
        Jump operators including their rates:
        L_k = sqrt(gamma_k) * J_k.
        Use an empty list for Hamiltonian-only evolution.
    H : array_like, shape (d, d)
        Hermitian Hamiltonian matrix.

    Returns
    -------
    drho : complex ndarray, shape (d, d)
        Time derivative of the density matrix, not the evolved state.

    Notes
    -----
    [H, ρ] = Hρ - ρH is the commutator.
    {A, ρ} = Aρ + ρA is the anticommutator.
    All matrices must use the same basis and dimension.
    """
    rho = np.asarray(rho, dtype=complex)
    H = np.asarray(H, dtype=complex)

    # Hamiltonian evolution
    drho = (-1j / hbar) * (H @ rho - rho @ H)

    # Dissipation
    for L in jump_operators:
        L = np.asarray(L, dtype=complex)
        drho += (L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L))
    return drho