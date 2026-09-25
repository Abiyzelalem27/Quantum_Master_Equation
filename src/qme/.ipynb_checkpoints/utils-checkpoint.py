

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


def lindblad_rhs(t, rho_flat, H, jumps):
    """
    Lindblad equation for a finite-dimensional system.

    H:     (d, d) Hamiltonian, with hbar = 1
    jumps: sequence of (d, d) jump operators
    """
    d = H.shape[0]
    rho = rho_flat.reshape(d, d)

    drho_dt = -1j * (H @ rho - rho @ H)

    for L in jumps:
        LdagL = L.conj().T @ L
        drho_dt += (
            L @ rho @ L.conj().T
            - 0.5 * (LdagL @ rho + rho @ LdagL)
        )

    return drho_dt.reshape(d * d)


def trace_environment(rho_SE):
    """Trace out a two-level environment; tensor order is S ⊗ E."""
    return np.trace(rho_SE.reshape(2, 2, 2, 2), axis1=1, axis2=3)


def joint_rhs(t, rho_flat, H_SE):
    """Return the flattened derivative -i[H_SE, rho_SE]."""
    rho_SE = rho_flat.reshape(4, 4)
    drho_SE_dt = -1j * (H_SE @ rho_SE - rho_SE @ H_SE)
    return drho_SE_dt.reshape(16)


def liouvillian_matrix(H, jumps):
    """
    Build L such that

        d vec(rho)/dt = L @ vec(rho)

    using NumPy's default row-major reshape order.
    """
    H = np.asarray(H, dtype=complex)
    d = H.shape[0]
    I = np.eye(d, dtype=complex)

    L_matrix = -1j * (
        np.kron(H, I) - np.kron(I, H.T)
    )

    for jump in jumps:
        jump = np.asarray(jump, dtype=complex)
        M = jump.conj().T @ jump

        L_matrix += (
            np.kron(jump, jump.conj())
            - 0.5 * np.kron(M, I)
            - 0.5 * np.kron(I, M.T)
        )

    return L_matrix