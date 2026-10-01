# Shor's Algorithm

Factor an odd composite number using Shor's algorithm on Amazon Braket —
the algorithm that, if run at scale on a large enough quantum computer,
would break RSA encryption. Runs against Braket's free local simulator
or a managed AWS simulator — see §7 for why real QPU hardware is a
harder story for this particular project than its two companions.

This is the third in a series of Braket "hello world" demos — after a
[Bell state](../bell) (qubits, kets, gates) and a [Grover's-algorithm
Magic 8-Ball](../quantum-eight-ball) (oracles, amplitude amplification).
This README assumes the qubit/ket/gate basics from the Bell-state
README and spends its length on what's new here: Quantum Phase
Estimation, the Quantum Fourier Transform, and the number theory that
turns "find a period" into "find a factor."

## 1. The problem, and why it matters

Given a composite number `N`, find a nontrivial factor. Classically,
the best known algorithms (e.g. the General Number Field Sieve) take
**sub-exponential** time in the number of digits of `N` — slow enough
that factoring a sufficiently large `N` (hundreds of digits) is
considered computationally infeasible. **RSA encryption's security
rests entirely on this assumed hardness**: the public key is built
from a large `N = p*q`, and recovering the private key requires
factoring `N`.

Shor's algorithm factors `N` in **polynomial** time on a quantum
computer. This isn't a modest speedup — it's the difference between
"infeasible for the lifetime of the universe" and "fast" for the key
sizes RSA actually uses, which is why Shor's algorithm is the
single most consequential result in quantum computing for real-world
cryptography (and why "post-quantum cryptography" — algorithms
believed hard even for quantum computers — is an active standardization
effort today). This project demonstrates the real algorithm on small
`N` (15, 21, ...) — nowhere near cryptographically relevant sizes, but
exactly the same math and the same quantum circuit structure.

## 2. From factoring to order-finding (all classical, so far)

Shor's insight was reducing factoring to a different problem — finding
the **multiplicative order** of a number — and that problem is where
the quantum computer actually comes in. Everything in this section is
ordinary number theory; no quantum mechanics yet.

**The order of `a` mod `N`** is the smallest positive integer `r` such
that:

```
a^r ≡ 1 (mod N)
```

(this is well-defined whenever `gcd(a, N) = 1`, by Euler's theorem —
some power of `a` always returns to 1 mod `N`).

**Why knowing `r` factors `N`.** Suppose `r` is even and
`a^(r/2) ≠ -1 (mod N)`. Then:

```
a^r - 1 ≡ 0 (mod N)
(a^(r/2) - 1)(a^(r/2) + 1) ≡ 0 (mod N)        [difference of squares]
```

So `N` divides the product `(a^(r/2)-1)(a^(r/2)+1)`, but (given our
two conditions) `N` does **not** divide either factor by itself —
`a^(r/2) ≢ 1` (since `r` is the *smallest* such exponent and `r/2 < r`)
and `a^(r/2) ≢ -1` (that's our second condition, ruled out explicitly).
A number that divides a product but divides neither factor must share
a **nontrivial common factor** with each one. That's exactly what
`gcd()` finds:

```
factor1 = gcd(a^(r/2) - 1, N)
factor2 = gcd(a^(r/2) + 1, N)
```

both guaranteed to be nontrivial (not `1`, not `N`) under those two
conditions. This is [`try_factor()`](shor.py) in the code, and it's
also exactly why those two conditions are checked before trusting any
candidate `r` — if `r` is odd, there's no integer `r/2` to work with;
if `a^(r/2) ≡ -1 (mod N)`, the second gcd is just `gcd(0, N) = N`
(trivial, not a real factor).

**How often does a random `a` work?** For `N` odd and not a prime
power, a theorem (not reproduced here, but standard in any Shor's
algorithm reference) guarantees that for a *uniformly random* `a`
coprime to `N`, the order `r` is even and `a^(r/2) ≠ -1 (mod N)` with
probability **at least 1/2**. So on average, two random choices of `a`
suffice — the real difficulty isn't finding a *good* `a`, it's
**computing `r` at all**. For large `N`, order-finding is itself
believed to be classically hard (closely related to the discrete
logarithm problem) — and that's precisely the subroutine a quantum
computer can do fast.

**One more classical step before any of this: check `gcd(a, N)`
directly.** If it's not `1`, you've already found a factor — no order,
no quantum computer, nothing further needed. [`classical_shortcut()`](shor.py)
checks this first, exactly as real implementations do, since it's free
and occasionally just hands you the answer (see §6's worked example).

## 3. Quantum Phase Estimation

**The general tool.** Given a unitary `U` and one of its eigenstates
`|u⟩` with eigenvalue `e^(2πiφ)` (for some unknown `φ ∈ [0,1)`),
**Quantum Phase Estimation (QPE)** estimates `φ`. It uses two
registers: a `t`-qubit **counting register** (starts at `|0...0⟩`) and
a **target register** holding `|u⟩`.

```
1. Apply H to every counting qubit: counting register -> uniform
   superposition over all t-bit strings x.
2. For counting qubit i (0 = most significant bit), apply U^(2^(t-1-i))
   to the target register, controlled on qubit i.
3. Apply the inverse Quantum Fourier Transform to the counting register.
4. Measure. The result y satisfies y/2^t ≈ φ.
```

**Why step 2's specific power of `U` per qubit matters.** After step 1,
the counting register is `(1/√2^t) Σₓ |x⟩` — every `t`-bit integer `x`
in superposition, with `x`'s binary digits exactly the counting
qubits' values (qubit 0 = most significant, same convention used
throughout this whole series of projects). Counting qubit `i`
controls `U^(2^(t-1-i))` — i.e. the power of 2 matching that qubit's
own positional value in binary. So across the whole register, the
*total* number of times `U` effectively gets applied, for a given `x`,
is `Σ (bit_i · 2^(t-1-i)) = x` — exactly `x` applications of `U`,
packaged as one `U^x`.

**The phase-kickback trick.** `U|u⟩ = e^(2πiφ)|u⟩`, so a
controlled-`U` applied with the target already in `|u⟩` doesn't change
`|u⟩` at all — it kicks the phase `e^(2πiφ)` back onto the *control*
qubit instead: `controlled-U|1⟩|u⟩ = e^(2πiφ)|1⟩|u⟩`. Doing this
conditionally, for every `x` simultaneously (since the counting
register is in superposition over all `x`), the counting register ends
up as:

```
(1/√2^t) Σₓ e^(2πiφx) |x⟩
```

This is a periodic function of `x` with "frequency" `φ` — and that's
precisely the kind of state the inverse QFT is built to read out (§4).
After the inverse QFT and a measurement, the result `y` satisfies
`y/2^t ≈ φ`, with the approximation becoming exact when `φ*2^t` is
already an integer, and otherwise concentrating most of its probability
near the nearest integers to it.

**Applying this to order-finding, without ever finding an eigenstate.**
We want to run QPE on `U|y⟩ = |a·y mod N⟩`, whose eigenvalues turn out
to be exactly `e^(2πis/r)` for `s = 0, ..., r-1` (`r` = the order we
want) — but finding an actual eigenstate `|uₛ⟩` of `U` sounds like it
requires already knowing `r`. The trick that makes Shor's algorithm
practical: **the state `|1⟩` is itself an equal superposition of every
eigenstate of `U`:**

```
|1⟩ = (1/√r) Σₛ |uₛ⟩
```

so just running QPE with the target register prepared in the easy,
`r`-independent state `|1⟩` (one `X` gate) is equivalent, by linearity,
to running it on a uniformly random `|uₛ⟩` each time. Every shot
samples some `s` uniformly from `0..r-1` and returns `y/2^t ≈ s/r` —
no diagonalization, no knowledge of `r` or the eigenstates required
ahead of time. This is exactly [`build_shor_circuit()`](shor.py):
`H` on the counting register, `X` on the target register's last qubit
(preparing `|1⟩`), the controlled-`U^(2^k)` ladder, inverse QFT.

## 4. The Quantum Fourier Transform

The QFT is the quantum analogue of the discrete Fourier transform. On
`t` qubits (`N = 2^t` basis states), it's the unitary matrix:

```
QFT[j,k] = (1/√N) · e^(2πi·j·k/N)
```

— identical in form to the classical DFT matrix, just acting on
quantum amplitudes instead of a classical signal. Shor's algorithm
needs its **inverse** (conjugate transpose, `QFT⁻¹[j,k] = (1/√N)·
e^(-2πi·j·k/N)`), applied in step 3 of QPE above.

**The circuit**, for `t` qubits `q₀...q_{t-1}` (forward direction):

```
for j in 0..t-1:
    H(qⱼ)
    for k in j+1..t-1:
        controlled-phase(control=q_k, target=qⱼ, angle=2π/2^(k-j+1))
swap(q_i, q_{t-1-i}) for i in 0..t/2-1   # reverse qubit order
```

The inverse QFT is this same gate sequence run with every individual
gate inverted (`H` is self-inverse, a phase gate's inverse negates its
angle, swap is self-inverse) **and in reverse order** — standard for
inverting any sequence of unitaries. [`qft_circuit()`](shor.py) and
[`inverse_qft_circuit()`](shor.py) implement both; both are verified in
[`test_shor.py`](test_shor.py) against the exact textbook DFT matrix
computed directly with `numpy`, not just by eyeballing the circuit —
QFT bit-ordering is a notoriously easy place to introduce an
off-by-one or a sign error, so this project checks it numerically
rather than trusting a derivation.

**Why the inverse QFT turns QPE's state into a measurable peak.**
Feed `(1/√2^t) Σₓ e^(2πiφx)|x⟩` (§3's post-controlled-`U` state) into
the inverse QFT. Writing out `QFT⁻¹`'s action and simplifying, the
amplitude on output state `|y⟩` becomes a geometric sum that is
largest when `y ≈ φ·2^t`, and — in the special case where `φ·2^t` is
exactly an integer — collapses to amplitude exactly `1` on that single
`y` and `0` everywhere else (a textbook geometric-series cancellation).
That's exactly what the worked example in §6 shows numerically: when
`r` divides `2^t` evenly, the measured phases land on *exact* peaks
with no spreading at all.

## 5. From a measured phase to the period: continued fractions

A single measurement gives `y`, and `y/2^t ≈ s/r` for the (unknown)
integer `s` that happened to come up this shot. We know `y` and `2^t`
exactly; we want `r`. [`Fraction(y, 2**t).limit_denominator(N)`](shor.py)
— Python's standard-library rational-number type — does exactly this:
it finds the fraction with the smallest denominator ≤ `N` that's
closest to `y/2^t`, via the continued-fractions algorithm. Since
`s/r` is itself such a fraction (with denominator `r ≤ N`), this
recovers `r` whenever `s` and `r` share no common factor.

**The catch — and why `try_factor()` verifies before trusting `r`.**
If `gcd(s, r) > 1`, the fraction `s/r` *reduces* — e.g. `2/4` reduces
to `1/2` — and `limit_denominator()` has no way to recover the
original, un-reduced `r` from the reduced fraction alone; it correctly
returns the *reduced* denominator, which is a proper divisor of the
true order, not the order itself. **This is a real, expected failure
mode of Shor's algorithm, not a bug** — some fraction of measurement
outcomes simply don't carry enough information to recover `r`, which
is exactly why the algorithm is run for multiple shots and why
`try_factor()` explicitly checks `a^r mod N == 1` before using a
candidate `r` at all, discarding anything that fails (even if, by
coincidence, plugging a wrong `r` into the gcd formula would have
*also* produced a correct factor — that happening on a technically
invalid `r` is luck, not something to rely on in general).

**Worked numeric example — this is a real, verified run of this
project**, for `N=15`, `a=7` (`t=8` counting qubits):

| measured `y` | phase `y/2⁸` | continued-fractions `r` | `a^r mod N` | valid? | factors |
|---|---|---|---|---|---|
| 0 | 0.0 | 1 | `7¹ mod 15 = 7` | no (≠1) | — |
| 64 | 0.25 | 4 | `7⁴ mod 15 = 1` | **yes** | `gcd(7²-1,15)=3`, `gcd(7²+1,15)=5` |
| 128 | 0.5 | 2 | `7² mod 15 = 4` | no (≠1) | — (would coincidentally give 3, but `r=2` isn't real — see above) |
| 192 | 0.75 | 4 | `7⁴ mod 15 = 1` | **yes** | same as above: `3, 5` |

Exactly 2 of the 4 possible outcomes (`y=64` and `y=192`, each ~25% of
shots) verify and yield the real factorization `15 = 3 × 5`; `y=0` is
the always-uninformative trivial case; `y=128`'s `r=2` fails
verification and is correctly discarded. That's roughly a 50% per-shot
success rate for this `(N, a)` pair — typical for Shor's algorithm,
and why `--shots` defaults to 20 rather than 1.

## 6. Running it: a full worked example

```bash
python shor.py --N 15 --a 7 --shots 20
```

```
Factoring N = 15 with base a = 7

Quantum phase estimation circuit (8 counting qubits + 4 work qubits = 12 total):
  (diagram omitted: 12 qubits, too wide to print legibly)

Running 20 shot(s) on <LocalSimulator>...
Task ID: 63cd42ce-03a7-470e-ae79-13ece36a2a9f

Measured phases (most frequent first):
  10000000 (y=128, phase~0.5000, r candidate= 2, count=6) -> no new factor
  00000000 (y=  0, phase~0.0000, r candidate= 1, count=6) -> no new factor
  11000000 (y=192, phase~0.7500, r candidate= 4, count=5) -> factors (3, 5)
  01000000 (y= 64, phase~0.2500, r candidate= 4, count=3) -> factors (3, 5)

Success: measured 11000000 gave r=4, and 15 = 3 x 5
```

That matches §5's table exactly: four distinct outcomes, two of which
verify and factor `15` as `3 × 5`.

**The classical-shortcut path**, which skips the quantum circuit
entirely:

```bash
python shor.py --N 15 --a 3
```
```
Factoring N = 15 with base a = 3

gcd(3, 15) = 3 -- that's already a factor! 15 = 3 x 5
(No quantum computer needed this time -- try a different --a to see the quantum path.)
```

(`3` and `15` aren't coprime, so this is the real first step of the
real algorithm working as intended — not a special case bypassed for
convenience.)

## 7. The code

[`shor.py`](shor.py):

- `n_bits_for(N)` — bits needed for the work register (`⌈log₂ N⌉`).
- `pick_base(N, rng)` / `classical_shortcut(a, N)` — §2's random base
  and the free gcd check.
- `try_factor(a, r, N)` — §2's `gcd(a^(r/2)±1, N)`, with the `a^r mod N
  == 1` verification from §5.
- `qft_circuit(qubits)` / `inverse_qft_circuit(qubits)` — §4's QFT, in
  both directions (only the inverse is used by `build_shor_circuit`;
  the forward version exists as the more commonly-taught reference
  point, and both are unit-tested against the exact DFT matrix).
- `modmul_unitary(multiplier, N, n_work)` — the permutation matrix for
  `|y⟩ ↦ |multiplier·y mod N⟩`.
- `controlled_unitary_matrix(U)` — wraps any unitary as a
  controlled version: identity when the control is `|0⟩`, `U` when
  it's `|1⟩`.
- `build_modexp_circuit(...)` — §3's controlled-`U^(2^k)` ladder, one
  gate per counting qubit, each built by computing `a^(2^k) mod N`
  classically (Python's `pow(a, 2**k, N)`, itself a repeated-squaring
  computation — a nice parallel to the quantum circuit's own
  repeated-squaring structure) and turning that single multiplier into
  one exact matrix via `modmul_unitary` + `controlled_unitary_matrix`.
- `build_shor_circuit(...)` — assembles the full circuit: `H` on the
  counting register, `X` to prepare `|1⟩`, the modular-exponentiation
  ladder, inverse QFT.
- `wait_for_result(task)` — same AWS task-ID/cancel-on-interrupt
  helper as the sibling projects.
- `main()` — CLI: pick/validate `a`, try the classical shortcut, build
  and run the circuit, walk every distinct measured outcome through
  continued fractions and `try_factor`, report the first success.

**On `Circuit.unitary()`.** Rather than hand-building a dedicated
arithmetic circuit (adders, controlled multipliers, out of elementary
gates) for *every* `(N, a)` pair — a substantial undertaking in its
own right, and the subject of real published papers — each
controlled-`U^(2^k)` is applied as one exact matrix via Braket's
generic `Circuit.unitary(matrix=..., targets=...)` gate. This keeps
the implementation's complexity independent of `N` and `a` entirely,
at a real cost: it only runs on simulators. See §8 for exactly why,
and what a QPU-compatible version would need instead.

## 8. Setup

Requires Python 3.9+ (tested with 3.9).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 8.1 Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

Covers: the QFT and inverse QFT against the exact DFT matrix (§4),
`modmul_unitary` being a valid permutation (unitary) matrix for various
multipliers, `controlled_unitary_matrix`'s block structure,
`try_factor` against known cases (including explicitly checking that
`N=15`'s spurious `r=2` is correctly rejected — see §5), the classical
shortcut, and an end-to-end run of the full circuit for `N=15, a=7`
checking that the measured distribution matches §6's table.

## 9. Running on AWS Braket

### 9.1 Managed simulators

```bash
python shor.py --N 15 --a 7 --device sv1 --shots 20
```

Same `local` / `sv1` / `dm1` / `tn1` / `qpu` device flag as the sibling
projects, same one-time AWS setup (credentials, S3 bucket — see the
[Bell-state README](../bell/README.md#51-one-time-aws-setup) for the
full steps). SV1's qubit limit (34) comfortably covers `N` well beyond
what's practical to interpret by hand.

### 9.2 Real quantum hardware (QPU) — doesn't work here, and why

Unlike its sibling projects, **this one cannot run on a real QPU as
written.** Braket's generic `Circuit.unitary()` gate — what
`build_modexp_circuit()` uses for every controlled-`U^(2^k)` — is a
simulator-only operation. Checked directly against a real device's
published capabilities (IonQ Forte-1's supported-operations list has
no `unitary` entry at all, only elementary gates like `x`, `y`, `z`,
`h`, native rotations, etc.), so `--device qpu` will fail outright
here, not just run noisily like the Grover project's larger `--qubits`
values do.

Building a real QPU-compatible version would mean replacing
`build_modexp_circuit()` with an actual **arithmetic circuit** —
quantum adders and controlled modular multipliers built from `CNOT`,
Toffoli, and QFT-based addition (e.g. the Draper adder), the subject
of real published circuit constructions for modular exponentiation.
That's a substantial project in its own right, well beyond converting
a flag — real demonstrations of Shor's algorithm on actual quantum
hardware (factoring 15, famously) have historically needed exactly
this kind of custom, N-specific circuit optimization, and even then
only work on today's hardware for the smallest textbook cases.

## 10. Extending this

- **Larger `N`.** Qubit count grows as `3 × ⌈log₂ N⌉` (work + counting
  registers), so `N` in the dozens to low hundreds stays comfortable
  on the local simulator; beyond that, state-vector memory becomes the
  limit (each extra qubit doubles it) well before anything about the
  algorithm itself changes. Try `--N 21`, `--N 33`, `--N 35`.
- **The classical pre-checks this project skips.** A fully rigorous
  implementation also checks `N` isn't even (trivial factor `2`) and
  isn't a perfect power `p^k` (via `k`-th root checks) before ever
  reaching the quantum step — both are standard, cheap classical tests
  real implementations include; this project just assumes you've
  passed in a suitable odd, non-prime-power `N`.
- **A genuinely QPU-compatible modular exponentiation circuit** — see
  §9.2.
