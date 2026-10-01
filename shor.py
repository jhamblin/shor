"""Shor's algorithm for integer factorization, on Amazon Braket.

Factors an odd composite N by:
  1. A classical shortcut: pick a base `a` and check gcd(a, N) first --
     if it's not 1, you've already found a factor for free.
  2. The quantum part: Quantum Phase Estimation (QPE) on the modular
     multiplication operator U|y> = |a*y mod N>, which reveals s/r for
     some integer s, where r is the multiplicative order of a mod N
     (the smallest r with a^r = 1 mod N).
  3. Classical post-processing: continued fractions recover r from the
     measured phase, then gcd(a^(r/2) +/- 1, N) gives the factors --
     whenever r is even and a^(r/2) != -1 mod N.

See README.md for the full math (why order-finding factors N, how QPE
and the Quantum Fourier Transform work, and a fully worked N=15
example) behind every step here.
"""

import argparse
import math
import os
import random
import sys
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np
from braket.circuits import Circuit
from braket.devices import LocalSimulator

DEFAULT_N = 15


def _load_quantum_arithmetic():
    """Import the sibling quantum-arithmetic repo, used only by
    --backend gates. Assumes both repos live as siblings under the same
    parent directory (no packaging/pip-install machinery exists in this
    project family) -- see quantum-arithmetic/README.md section 7."""
    sibling = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantum-arithmetic")
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    try:
        import quantum_arithmetic
    except ImportError as e:
        raise SystemExit(
            "--backend gates requires the quantum-arithmetic sibling repo, "
            f"expected at {os.path.normpath(sibling)}. Clone it next to this "
            "one (see quantum-arithmetic's README)."
        ) from e
    return quantum_arithmetic

MANAGED_SIMULATORS = {
    "sv1": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    "dm1": "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
    "tn1": "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
}


def get_device(name: str, qpu_arn: Optional[str]):
    if name == "local":
        return LocalSimulator()

    # Imported lazily so `--device local` never requires AWS credentials.
    from braket.aws import AwsDevice

    if name == "qpu":
        if not qpu_arn:
            raise SystemExit("--qpu-arn is required when --device qpu")
        return AwsDevice(qpu_arn)

    return AwsDevice(MANAGED_SIMULATORS[name])


def wait_for_result(task):
    """Wait for a submitted quantum task's result, printing its ID so it
    can be found or cancelled manually even if this script is interrupted.

    See the Bell-state and quantum-eight-ball sibling projects for the
    same helper -- Ctrl-C only stops local polling, not the task itself,
    so this requests cancellation on AWS too (best-effort for QPUs).
    """
    print(f"Task ID: {task.id}")
    try:
        return task.result()
    except KeyboardInterrupt:
        print("\nInterrupted -- requesting cancellation on AWS...")
        try:
            task.cancel()
            print(
                f"Cancellation requested for {task.id}. If the task had "
                "already started running, it may complete (and be billed) "
                "anyway -- check its status in the Braket console."
            )
        except Exception as e:
            print(f"Could not cancel: {e}")
        raise SystemExit(1)


# --- Number theory -----------------------------------------------------


def n_bits_for(N: int) -> int:
    """Smallest n such that 2**n >= N (bits needed for residues 0..N-1)."""
    return max(1, (N - 1).bit_length())


def pick_base(N: int, rng: random.Random) -> int:
    """Pick a random base in [2, N-2]. Does NOT check gcd -- that's
    classical_shortcut()'s job, since finding a non-coprime base here
    is itself a (lucky) classical solution, not something to avoid."""
    return rng.randrange(2, N - 1)


def classical_shortcut(a: int, N: int) -> Optional[int]:
    """If gcd(a, N) > 1, that gcd is already a nontrivial factor of N --
    no quantum computer needed. Real implementations of Shor's algorithm
    always check this first."""
    g = math.gcd(a, N)
    return g if 1 < g < N else None


def try_factor(a: int, r: int, N: int) -> Optional[Tuple[int, int]]:
    """Given a candidate order r (from continued fractions), return a
    nontrivial factor pair of N if r actually works, else None.

    Critically, this verifies a^r == 1 mod N before trusting r at all.
    Continued fractions can return a proper divisor of the true order
    (when the measured phase s/r reduces because gcd(s, r) > 1) that
    does NOT itself satisfy a^r == 1 -- using such an r anyway is not
    mathematically justified, even though it can occasionally still
    stumble onto a correct factor by coincidence. See README section 5.
    """
    if r == 0 or r % 2 != 0:
        return None
    if pow(a, r, N) != 1:
        return None
    x = pow(a, r // 2, N)
    if x == N - 1:  # a^(r/2) == -1 mod N: the gcd step would be trivial
        return None
    f1, f2 = math.gcd(x - 1, N), math.gcd(x + 1, N)
    nontrivial = sorted({f for f in (f1, f2) if 1 < f < N})
    if len(nontrivial) >= 2:
        return nontrivial[0], nontrivial[1]
    if len(nontrivial) == 1:
        return nontrivial[0], N // nontrivial[0]
    return None


# --- Quantum Fourier Transform -------------------------------------------


def qft_circuit(qubits: List[int]) -> Circuit:
    """Forward QFT. Not used by Shor's algorithm directly (it uses the
    inverse, below) but included since it's the more commonly taught
    direction and the natural reference point for inverse_qft_circuit."""
    t = len(qubits)
    circuit = Circuit()
    for j in range(t):
        circuit.h(qubits[j])
        for k in range(j + 1, t):
            circuit.cphaseshift(qubits[k], qubits[j], 2 * math.pi / (2 ** (k - j + 1)))
    for i in range(t // 2):
        circuit.swap(qubits[i], qubits[t - 1 - i])
    return circuit


def inverse_qft_circuit(qubits: List[int]) -> Circuit:
    """Inverse QFT: swap-then-rotate, in reverse order with negated
    angles from qft_circuit -- the inverse of a gate sequence is the
    reversed sequence of each gate's own inverse."""
    t = len(qubits)
    circuit = Circuit()
    for i in range(t // 2):
        circuit.swap(qubits[i], qubits[t - 1 - i])
    for j in reversed(range(t)):
        for k in reversed(range(j + 1, t)):
            circuit.cphaseshift(qubits[k], qubits[j], -2 * math.pi / (2 ** (k - j + 1)))
        circuit.h(qubits[j])
    return circuit


# --- Modular exponentiation -----------------------------------------------


def modmul_unitary(multiplier: int, N: int, n_work: int) -> np.ndarray:
    """Permutation matrix for |y> -> |multiplier*y mod N> on y in 0..N-1;
    basis states >= N (padding up to 2**n_work) map to themselves."""
    dim = 2**n_work
    U = np.zeros((dim, dim))
    for y in range(dim):
        new_y = (y * multiplier) % N if y < N else y
        U[new_y, y] = 1
    return U


def controlled_unitary_matrix(U: np.ndarray) -> np.ndarray:
    """Block-diagonal controlled-U: identity when the control is |0>, U
    when the control is |1>. The control is the matrix's first (most
    significant) qubit -- see build_modexp_circuit for how this lines
    up with Circuit.unitary()'s `targets` ordering."""
    dim = U.shape[0]
    CU = np.eye(2 * dim)
    CU[dim:, dim:] = U
    return CU


def build_modexp_circuit(
    a: int, N: int, n_work: int, counting_qubits: List[int], work_qubits: List[int]
) -> Circuit:
    """The controlled-U^(2^k) ladder of Quantum Phase Estimation, with
    U|y> = |a*y mod N>. Counting qubit i (0 = most significant) controls
    U^(2^(t-1-i)), so that the counting register's bits, read as a
    standard binary integer x, apply U^x in total -- see README section
    3 for the full derivation of why this is the right power per qubit.

    Each controlled-U^(2^k) is applied as one exact unitary matrix via
    Circuit.unitary() rather than decomposed into elementary gates. This
    keeps the implementation complexity-independent of N and a (no
    custom arithmetic circuit needed per factoring target) but means
    this part of the circuit -- unlike the QFT -- only runs on
    simulators, not real QPUs; see README section 7 for why that's a
    reasonable trade for a project at this scale.
    """
    t = len(counting_qubits)
    circuit = Circuit()
    for i, c_qubit in enumerate(counting_qubits):
        power = pow(a, 2 ** (t - 1 - i), N)
        U = modmul_unitary(power, N, n_work)
        CU = controlled_unitary_matrix(U)
        circuit.unitary(matrix=CU, targets=[c_qubit] + work_qubits)
    return circuit


def build_shor_circuit(
    a: int,
    N: int,
    counting_qubits: List[int],
    work_qubits: List[int],
    backend: str = "unitary",
    acc_qubits: Optional[List[int]] = None,
    ancilla: Optional[int] = None,
    and_ancilla: Optional[int] = None,
) -> Circuit:
    """backend="unitary" (default): build_modexp_circuit's Circuit.unitary()
    construction -- simulator-only, independent of N/a.
    backend="gates": the quantum-arithmetic sibling repo's modexp_ladder --
    elementary gates only, submittable to Rigetti/IQM (not IonQ -- see
    quantum-arithmetic/README.md section 8). Needs acc_qubits (width
    len(work_qubits)+1), ancilla, and and_ancilla -- see README section 9.2.
    """
    circuit = Circuit()
    for q in counting_qubits:
        circuit.h(q)
    circuit.x(work_qubits[-1])  # prepare the work register in state |1>
    if backend == "unitary":
        circuit.add_circuit(build_modexp_circuit(a, N, len(work_qubits), counting_qubits, work_qubits))
    elif backend == "gates":
        qa = _load_quantum_arithmetic()
        qa.modexp_ladder(circuit, counting_qubits, work_qubits, acc_qubits, ancilla, and_ancilla, a, N)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    circuit.add_circuit(inverse_qft_circuit(counting_qubits))
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--N", type=int, default=DEFAULT_N, help=f"Odd composite number to factor (default: {DEFAULT_N})."
    )
    parser.add_argument(
        "--a",
        type=int,
        default=None,
        help="Base for modular exponentiation. Random (coprime-checked) if omitted.",
    )
    parser.add_argument(
        "--counting-qubits",
        type=int,
        default=None,
        help="Size of the QPE counting register. Defaults to 2x the number "
        "of bits needed to represent N (the standard theoretical "
        "guarantee for Shor's algorithm's success probability).",
    )
    parser.add_argument(
        "--device",
        choices=["local", *MANAGED_SIMULATORS, "qpu"],
        default="local",
        help="Where to run the circuit (default: local). --backend unitary "
        "(the default) does not run on any real QPU; --backend gates runs "
        "on Rigetti/IQM but not IonQ -- see README section 9.",
    )
    parser.add_argument(
        "--backend",
        choices=["unitary", "gates"],
        default="unitary",
        help="How to build modular exponentiation. 'unitary' (default): "
        "Circuit.unitary(), simulator-only. 'gates': elementary gates "
        "from the quantum-arithmetic sibling repo, submittable to "
        "Rigetti/IQM (not IonQ) -- see README section 9.2.",
    )
    parser.add_argument("--qpu-arn", default=None, help="Device ARN to use when --device qpu.")
    parser.add_argument(
        "--shots",
        type=int,
        default=20,
        help="Measurement shots (default: 20). Each shot gives one phase "
        "estimate; only some lead to a usable period -- see README section 5.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for picking a random --a.")
    args = parser.parse_args()

    if args.N < 3 or args.N % 2 == 0:
        raise SystemExit("--N must be an odd number >= 3")

    rng = random.Random(args.seed)
    n_work = n_bits_for(args.N)
    n_counting = args.counting_qubits or 2 * n_work
    counting_qubits = list(range(n_counting))
    work_qubits = list(range(n_counting, n_counting + n_work))

    acc_qubits = ancilla = and_ancilla = None
    extra_qubits = 0
    if args.backend == "gates":
        acc_width = n_work + 1
        acc_qubits = list(range(n_counting + n_work, n_counting + n_work + acc_width))
        ancilla = n_counting + n_work + acc_width
        and_ancilla = ancilla + 1
        extra_qubits = acc_width + 2  # acc_qubits + ancilla + and_ancilla

    a = args.a if args.a is not None else pick_base(args.N, rng)
    print(f"Factoring N = {args.N} with base a = {a}")

    shortcut = classical_shortcut(a, args.N)
    if shortcut:
        print(
            f"\ngcd({a}, {args.N}) = {shortcut} -- that's already a factor! "
            f"{args.N} = {shortcut} x {args.N // shortcut}"
        )
        print("(No quantum computer needed this time -- try a different --a to see the quantum path.)")
        return

    device = get_device(args.device, args.qpu_arn)
    circuit = build_shor_circuit(
        a,
        args.N,
        counting_qubits,
        work_qubits,
        backend=args.backend,
        acc_qubits=acc_qubits,
        ancilla=ancilla,
        and_ancilla=and_ancilla,
    )
    total_qubits = n_counting + n_work + extra_qubits
    print(
        f"\nQuantum phase estimation circuit ({n_counting} counting qubits + "
        f"{n_work} work qubits"
        + (f" + {extra_qubits} ancilla ({args.backend} backend)" if extra_qubits else "")
        + f" = {total_qubits} total):"
    )
    if total_qubits <= 6:
        print(circuit)
    else:
        print(f"  (diagram omitted: {total_qubits} qubits, too wide to print legibly)")

    print(f"\nRunning {args.shots} shot(s) on {device}...")
    result = wait_for_result(device.run(circuit, shots=args.shots))
    full_counts = result.measurement_counts

    counting_counts = {}
    for bitstring, count in full_counts.items():
        counting_bits = bitstring[:n_counting]
        counting_counts[counting_bits] = counting_counts.get(counting_bits, 0) + count

    print("\nMeasured phases (most frequent first):")
    found = None
    for bits, count in sorted(counting_counts.items(), key=lambda kv: -kv[1]):
        y = int(bits, 2)
        phase = y / 2**n_counting
        r_candidate = Fraction(y, 2**n_counting).limit_denominator(args.N).denominator
        factors = try_factor(a, r_candidate, args.N)
        status = f"-> factors {factors}" if factors else "-> no new factor"
        print(f"  {bits} (y={y:>3}, phase~{phase:.4f}, r candidate={r_candidate:>2}, count={count}) {status}")
        if factors and found is None:
            found = (bits, r_candidate, factors)

    print()
    if found:
        bits, r, factors = found
        print(f"Success: measured {bits} gave r={r}, and {args.N} = {factors[0]} x {factors[1]}")
    else:
        print(
            f"No shot yielded a usable period for a={a}. This happens "
            "(see README section 5) -- try --shots with a higher count, "
            "a different --seed, or a different --a."
        )


if __name__ == "__main__":
    main()
