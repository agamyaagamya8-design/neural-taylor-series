from sympy import symbols, sin, cos, exp, series
import random

x = symbols('x')

functions = [
    sin(x),
    cos(x),
    exp(x),
    x**2 + 3*x + 1,
    x**3 - x + 2
]

def generate_sample():
    f = random.choice(functions)
    s = series(f, x, 0, 6)  # Taylor series up to x^5
    return str(f), str(s)

def generate_dataset(n=100):
    data = []
    for _ in range(n):
        inp, out = generate_sample()
        data.append((inp, out))
    return data

if __name__ == "__main__":
    dataset = generate_dataset(10)
    for i in dataset:
        print(i)
