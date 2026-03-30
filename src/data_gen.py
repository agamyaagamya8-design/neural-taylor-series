# src/data_gen.py
import random
from sympy import symbols, sin, cos, exp, series

x = symbols('x')
num_samples = 1000  
expansion_points = [0, 1, -1, 2, -2]  

functions = [
    'x',
    'x**2',
    'x**3',
    'x + 1',
    'x**2 + 3*x + 1',
    'x**3 - x + 2',
    'sin(x)',
    'cos(x)',
    'exp(x)',
    'sin(x) + x',
    'cos(x) - x**2',
    'exp(x) + x**3'
]

dataset = []
for _ in range(num_samples):
    func_str = random.choice(functions)
    func = eval(func_str)
    point = random.choice(expansion_points)
    try:
        taylor = str(series(func, x, point, 6))  # 5th order expansion
    except Exception as e:
        taylor = func_str 
    dataset.append((func_str, taylor))


import os
os.makedirs("data", exist_ok=True)  

with open("data/dataset.txt", "w") as f:
    for func, taylor in dataset:
        f.write(f"{func} -> {taylor}\n")
if __name__ == "__main__":
    dataset = generate_dataset(5000) 

    with open("data/dataset.txt", "w") as f:
        for inp, out in dataset:
            f.write(f"{inp} -> {out}\n")

    print("Dataset saved to data/dataset.txt")