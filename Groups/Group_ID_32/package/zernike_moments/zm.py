import numpy as np
from math import factorial

def radialpoly(r, n, m):
    rad = np.zeros(r.shape, r.dtype)
    P = (n - abs(m)) // 2
    Q = (n + abs(m)) // 2
    for s in range(P + 1):
        c = (-1) ** s * factorial(n - s)
        c /= factorial(s) * factorial(Q - s) * factorial(P - s)
        rad += c * r ** (n - 2 * s)
    return rad

def Zernikemoment(src, n, m):
    if src.dtype != np.float32:
        src = np.where(src > 0, 0, 1).astype(np.float32)
    if len(src.shape) == 3:
        print('the input image src should be in gray')
        return

    H, W = src.shape
    if H > W:
        src = src[(H - W) / 2: (H + W) / 2, :]
    elif H < W:
        src = src[:, (W - H) / 2: (H + W) / 2]

    N = src.shape[0]
    if N % 2:
        src = src[:-1, :-1]
        N -= 1
    x = range(N)
    y = x
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((2 * X - N + 1) ** 2 + (2 * Y - N + 1) ** 2) / N
    Theta = np.arctan2(N - 1 - 2 * Y, 2 * X - N + 1)
    R = np.where(R <= 1, 1, 0) * R

    # get the radial polynomial
    Rad = radialpoly(R, n, m)

    Product = src * Rad * np.exp(-1j * m * Theta)
    # calculate the moments
    Z = Product.sum()

    # count the number of pixels inside the unit circle
    cnt = np.count_nonzero(R) + 1
    # normalize the amplitude of moments
    Z = (n + 1) * Z / cnt
    # calculate the amplitude of the moment
    A = abs(Z)
    # calculate the phase of the mement (in degrees)
    Phi = np.angle(Z) * 180 / np.pi

    return Z, A, Phi
