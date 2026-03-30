"""
datagen.py — Dataset Generator for Neural Taylor Series
========================================================
Generates a dataset of (expression -> Taylor series expansion) pairs
using SymPy's symbolic math engine.

Output format (data/dataset.txt):
    sin(x) -> x - x**3/6 + x**5/120
    cos(x) -> 1 - x**2/2 + x**4/24
    ...

Hyperparameters / Configuration:
    TAYLOR_ORDER : int — Number of terms in expansion (default: 6)
    OUTPUT_FILE  : str — Path to save the dataset
    SEED         : int — Random seed for reproducibility
"""

import os
import random
from sympy import (
    symbols, sin, cos, exp, tan, log, sqrt,
    series, simplify, Rational
)

# ── Configuration ──────────────────────────────────────────────────────────────
TAYLOR_ORDER = 6          # truncate after this many terms (x^0 … x^(ORDER-1))
OUTPUT_FILE  = "data/dataset.txt"
SEED         = 42
# ───────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
x = symbols('x')


def get_functions():
    """
    Returns a list of (label, sympy_expression) tuples.
    Extend this list to enrich the dataset.
    """
    base = [
        # Trig
        ("sin(x)",        sin(x)),
        ("cos(x)",        cos(x)),
        ("tan(x)",        tan(x)),
        # Exponential / log
        ("exp(x)",        exp(x)),
        ("exp(-x)",       exp(-x)),
        ("log(1+x)",      log(1 + x)),
        ("log(1-x)",      log(1 - x)),
        # Polynomials
        ("x**2",          x**2),
        ("x**3",          x**3),
        ("x**4",          x**4),
        ("x**2 + x",      x**2 + x),
        ("x**3 - x",      x**3 - x),
        # Combinations
        ("sin(x)*exp(x)", sin(x) * exp(x)),
        ("cos(x)*exp(x)", cos(x) * exp(x)),
        ("sin(x)**2",     sin(x)**2),
        ("cos(x)**2",     cos(x)**2),
        # Square root / rational
        ("sqrt(1+x)",     sqrt(1 + x)),
        ("1/(1+x)",       1 / (1 + x)),
        ("1/(1-x)",       1 / (1 - x)),
        ("1/(1+x**2)",    1 / (1 + x**2)),
        # Scaled versions for variety
        ("sin(2*x)",      sin(2 * x)),
        ("cos(2*x)",      cos(2 * x)),
        ("exp(x**2)",     exp(x**2)),
        ("sin(x)/x",      sin(x) / x),   # sinc-like (series handles this fine)
    ]
    return base


def compute_taylor(expr, order: int = TAYLOR_ORDER) -> str:
    """
    Compute the Taylor series of expr around x=0, truncated at x^(order-1),
    and return it as a cleaned string.
    """
    try:
        s = series(expr, x, 0, order).removeO()
        return str(s)
    except Exception as e:
        return None   # skip expressions that fail


def generate_dataset(output_file: str = OUTPUT_FILE, order: int = TAYLOR_ORDER):
    """
    Main entry point: compute Taylor series for all functions and write to file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    functions = get_functions()
    pairs = []

    print(f"Generating Taylor series (order={order}) for {len(functions)} expressions …")

    for label, expr in functions:
        taylor_str = compute_taylor(expr, order)
        if taylor_str is None:
            print(f"  ⚠ Skipping '{label}' — SymPy could not compute series.")
            continue
        pairs.append((label, taylor_str))
        print(f"  ✓  {label:25s} ->  {taylor_str}")

    # Shuffle for good measure (reproducible via SEED)
    random.shuffle(pairs)

    with open(output_file, "w") as f:
        for label, taylor_str in pairs:
            f.write(f"{label} -> {taylor_str}\n")

    print(f"\nDataset saved to '{output_file}'  ({len(pairs)} pairs)")
    return pairs


if __name__ == "__main__":
    generate_dataset()
