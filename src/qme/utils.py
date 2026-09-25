

import numpy as np

def apply_kraus(rho, kraus):
    """Apply E(rho) = sum_k K_k rho K_k†."""
    return sum(K @ rho @ K.conj().T for K in kraus)


def trace_preserving(kraus, atol=1e-10):
    """TP iff sum_k K_k† K_k = I."""
    d = kraus[0].shape[1]
    completeness = sum(K.conj().T @ K for K in kraus)
    is_tp = np.allclose(completeness, np.eye(d), atol=atol)
    return is_tp


def choi_matrix(channel, d):
    """
    J(E) = sum_ij |i><j| ⊗ E(|i><j|).

    This unnormalized Choi matrix has trace d for a TP map.
    """
    J = np.zeros((d * d, d * d), dtype=complex)

    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d), dtype=complex)
            Eij[i, j] = 1.0
            J += np.kron(Eij, channel(Eij))

    return J


def completely_positive(channel, d, atol=1e-10):
    """CP iff the Choi matrix is positive semidefinite."""
    J = choi_matrix(channel, d)
    eigenvalues = np.linalg.eigvalsh(J)
    is_cp = bool(np.all(eigenvalues >= -atol))
    
    return is_cp


def hermiticity(rho, channel, atol=1e-10):
    """
    Check that this Hermitian input gives a Hermitian output.

    This is a check on the chosen input, not a proof for every input.
    """
    output = channel(rho)
    is_hermitian = (
        np.allclose(rho, rho.conj().T, atol=atol)
        and np.allclose(output, output.conj().T, atol=atol)
    )

    return is_hermitian