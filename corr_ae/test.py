import numpy as np

a = np.array([[0.1, 0.5], [-0.6, -0.9]])


def change_a(x):
    v = x.copy()
    v[v >= 0] = 1
    v[v != 1] = 0

def show_a(x):
    print(x)

change_a(a)
show_a(a)
