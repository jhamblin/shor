"""Tests for shor.py.

Run with:
    pip install -r requirements-dev.txt
    pytest

Covers: the QFT and inverse QFT against the exact textbook DFT matrix
(README section 4 -- bit-ordering bugs here are easy to introduce and
easy to miss without a numeric check), modmul_unitary/controlled_unitary_matrix
being genuinely valid (unitary) matrices, try_factor against known
cases (including the N=15 spurious-r=2 rejection from README section
5), the classical shortcut, and an end-to-end run of the full circuit.
"""

import math

import numpy as np
import pytest
from braket.circuits import Circuit
from braket.devices import LocalSimulator

import shor


def dft_matrix(t: int) -> np.ndarray:
    dim = 2**t
    j = np.arange(dim).reshape(-1, 1)
    k = np.arange(dim).reshape(1, -1)
    return np.exp(2j * np.pi * j * k / dim) / np.sqrt(dim)


# --- Quantum Fourier Transform, README section 4 --------------------------


@pytest.mark.parametrize("t", [1, 2, 3, 4])
def test_qft_matches_forward_dft(t):
    circuit = shor.qft_circuit(list(range(t)))
    assert np.allclose(circuit.to_unitary(), dft_matrix(t))


@pytest.mark.parametrize("t", [1, 2, 3, 4])
def test_inverse_qft_matches_conjugate_dft(t):
    circuit = shor.inverse_qft_circuit(list(range(t)))
    assert np.allclose(circuit.to_unitary(), dft_matrix(t).conj())


@pytest.mark.parametrize("t", [2, 3])
def test_qft_and_inverse_qft_compose_to_identity(t):
    qubits = list(range(t))
    circuit = Circuit().add_circuit(shor.qft_circuit(qubits)).add_circuit(
        shor.inverse_qft_circuit(qubits)
    )
    assert np.allclose(circuit.to_unitary(), np.eye(2**t))


# --- Modular exponentiation, README section 3 ------------------------------


@pytest.mark.parametrize("N,n_work", [(15, 4), (21, 5), (35, 6)])
def test_modmul_unitary_is_a_valid_permutation(N, n_work):
    """A 0/1 matrix is unitary (here, orthogonal) iff it's a permutation
    matrix: exactly one 1 per row and per column. Checked directly rather
    than via U @ U.T == I, which triggers a spurious (values are exactly
    correct either way) RuntimeWarning from a non-contiguous-transpose
    BLAS path on some platforms."""
    for multiplier in range(1, N):
        if math.gcd(multiplier, N) != 1:
            continue
        U = shor.modmul_unitary(multiplier, N, n_work)
        assert np.array_equal(U.sum(axis=0), np.ones(2**n_work)), f"multiplier={multiplier}"
        assert np.array_equal(U.sum(axis=1), np.ones(2**n_work)), f"multiplier={multiplier}"
        assert set(np.unique(U)) <= {0.0, 1.0}, f"multiplier={multiplier}"


def test_modmul_unitary_implements_multiply_mod_n():
    N, n_work = 15, 4
    U = shor.modmul_unitary(7, N, n_work)
    for y in range(N):
        column = U[:, y]
        expected = (7 * y) % N
        assert np.argmax(column) == expected
        assert column[expected] == 1


def test_modmul_unitary_fixes_padding_states():
    # 2**4 = 16 > N = 15, so basis state 15 isn't a real residue and
    # must map to itself (an arbitrary but fixed, unitary-preserving choice).
    U = shor.modmul_unitary(7, 15, 4)
    assert U[15, 15] == 1


def test_controlled_unitary_matrix_structure():
    U = shor.modmul_unitary(7, 15, 4)
    CU = shor.controlled_unitary_matrix(U)
    dim = U.shape[0]
    assert np.allclose(CU[:dim, :dim], np.eye(dim))  # control=0 -> identity
    assert np.allclose(CU[dim:, dim:], U)  # control=1 -> U
    assert np.allclose(CU[:dim, dim:], 0)
    assert np.allclose(CU[dim:, :dim], 0)


def test_build_modexp_circuit_applies_correct_power_per_qubit():
    """Directly checks the controlled-U^(2^(t-1-i)) power assignment from
    README section 3, by preparing the counting register to a known
    integer x and the work register to |1>, then checking the work
    register ends at a^x mod N (exactly as if U were applied x times)."""
    N, a = 15, 7
    n_work = shor.n_bits_for(N)
    t = 3
    counting = list(range(t))
    work = list(range(t, t + n_work))

    for x in range(2**t):
        circuit = Circuit()
        bits = format(x, f"0{t}b")
        for q, bit in zip(counting, bits):
            if bit == "1":
                circuit.x(q)
        circuit.x(work[-1])  # prepare |1>
        circuit.add_circuit(shor.build_modexp_circuit(a, N, n_work, counting, work))

        result = LocalSimulator().run(circuit, shots=1).result()
        bitstring = next(iter(result.measurement_counts))
        work_value = int(bitstring[t:], 2)
        assert work_value == pow(a, x, N), f"x={x}: expected {pow(a, x, N)}, got {work_value}"


# --- Classical number theory, README section 2 -----------------------------


def test_n_bits_for():
    assert shor.n_bits_for(15) == 4
    assert shor.n_bits_for(16) == 4
    assert shor.n_bits_for(17) == 5
    assert shor.n_bits_for(2) == 1


def test_classical_shortcut_finds_shared_factor():
    assert shor.classical_shortcut(3, 15) == 3
    assert shor.classical_shortcut(5, 15) == 5


def test_classical_shortcut_none_when_coprime():
    assert shor.classical_shortcut(7, 15) is None
    assert shor.classical_shortcut(2, 15) is None


@pytest.mark.parametrize(
    "a,r,N,expected",
    [
        (7, 4, 15, (3, 5)),  # the worked example from README section 5
        (7, 2, 15, None),  # spurious r from a reduced fraction -- must be rejected
        (7, 1, 15, None),  # trivial r=1
        (7, 3, 15, None),  # odd r
        (2, 6, 21, (3, 7)),
    ],
)
def test_try_factor_known_cases(a, r, N, expected):
    result = shor.try_factor(a, r, N)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert set(result) == set(expected)
        assert N % result[0] == 0 and N % result[1] == 0


def test_try_factor_rejects_r_that_fails_verification():
    """The core correctness property from README section 5: a candidate r
    that doesn't actually satisfy a^r == 1 mod N must never be used, even
    though plugging it into the gcd formula anyway can occasionally still
    produce a number that happens to divide N."""
    a, r, N = 7, 2, 15
    assert pow(a, r, N) != 1  # confirms r=2 is genuinely spurious here
    assert shor.try_factor(a, r, N) is None


# --- End-to-end, README section 6 ------------------------------------------


def test_end_to_end_factors_15():
    N, a = 15, 7
    n_work = shor.n_bits_for(N)
    t = 2 * n_work
    counting = list(range(t))
    work = list(range(t, t + n_work))

    circuit = shor.build_shor_circuit(a, N, counting, work)
    result = LocalSimulator().run(circuit, shots=200).result()
    counts = result.measurement_counts

    found_factors = False
    for bitstring, _count in counts.items():
        y = int(bitstring[:t], 2)
        from fractions import Fraction

        r = Fraction(y, 2**t).limit_denominator(N).denominator
        if shor.try_factor(a, r, N):
            found_factors = True
            break
    assert found_factors


def test_end_to_end_factors_15_with_gates_backend():
    """Same as test_end_to_end_factors_15 but via --backend gates (the
    quantum-arithmetic sibling repo's elementary-gate modexp_ladder,
    instead of Circuit.unitary()) -- skipped if that sibling repo isn't
    cloned alongside this one, since it's not a pip dependency (see
    README section 9.2)."""
    try:
        shor._load_quantum_arithmetic()
    except SystemExit:
        pytest.skip("quantum-arithmetic sibling repo not found; see README section 9.2")
    N, a = 15, 7
    n_work = shor.n_bits_for(N)
    t = 2 * n_work
    counting = list(range(t))
    work = list(range(t, t + n_work))
    acc_width = n_work + 1
    acc = list(range(t + n_work, t + n_work + acc_width))
    ancilla = t + n_work + acc_width
    and_ancilla = ancilla + 1

    circuit = shor.build_shor_circuit(
        a, N, counting, work, backend="gates", acc_qubits=acc, ancilla=ancilla, and_ancilla=and_ancilla
    )
    result = LocalSimulator().run(circuit, shots=200).result()
    counts = result.measurement_counts

    found_factors = False
    for bitstring, _count in counts.items():
        y = int(bitstring[:t], 2)
        from fractions import Fraction

        r = Fraction(y, 2**t).limit_denominator(N).denominator
        if shor.try_factor(a, r, N):
            found_factors = True
            break
    assert found_factors


def test_build_shor_circuit_rejects_unknown_backend():
    with pytest.raises(ValueError):
        shor.build_shor_circuit(7, 15, [0, 1], [2, 3], backend="nonsense")


def test_get_device_local_needs_no_aws_credentials():
    device = shor.get_device("local", None)
    assert isinstance(device, LocalSimulator)


def test_get_device_qpu_requires_an_arn():
    with pytest.raises(SystemExit):
        shor.get_device("qpu", None)


# --- wait_for_result ---------------------------------------------------


def test_wait_for_result_returns_normally_without_interrupt():
    class FakeTask:
        id = "fake-id"

        def result(self):
            return "the result"

    assert shor.wait_for_result(FakeTask()) == "the result"


def test_wait_for_result_cancels_task_on_keyboard_interrupt():
    class FakeTask:
        id = "arn:aws:braket:us-west-2:123:quantum-task/fake"
        cancel_called = False

        def result(self):
            raise KeyboardInterrupt

        def cancel(self):
            self.cancel_called = True

    task = FakeTask()
    with pytest.raises(SystemExit):
        shor.wait_for_result(task)
    assert task.cancel_called


def test_wait_for_result_raises_clearly_on_failed_task():
    """Regression test: task.result() returns None (not an exception) when
    a task fails -- the circumstance that actually caused this script to
    crash with a confusing AttributeError on a real Rigetti Cepheus-1-108Q
    run, when --backend gates's circuit exceeded the device compiler's
    gate-count limit (see README section 9.2) -- instead of a clear message."""

    class FakeTask:
        id = "fake-id"

        def result(self):
            return None

        def state(self):
            return "FAILED"

    with pytest.raises(SystemExit, match="FAILED"):
        shor.wait_for_result(FakeTask())
