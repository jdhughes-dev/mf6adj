import numpy as np
import scipy.special as sc
from scipy.optimize import fsolve


def multichoose(n,k):
    if k < 0 or n < 0: return "Error"
    if not k: return [[0]*n]
    if not n: return []
    if n == 1: return [[k]]
    return [[0]+val for val in multichoose(n-1,k)] + \
        [[val[0]+1]+val[1:] for val in multichoose(n,k-1)]

def multinomial(lst):
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0+1:]:
        for j in range(1,a+1):
            res *= i
            res //= j
            i -= 1
    return res

def Kr(l, m, TH):
    a = TH ** l
    b = (1 - TH ** (1/m)) ** m
    return a * (1 - b) ** 2


def integral(a, b, m, x):
    a1 = m * (1 - a) - 1
    b1 = m * (1 - b)
    c1 = m * (1 - a)
    y = x ** (1 / m)
    y1 = x ** (a1 / m)
    c = m / a1
    # print(a1, b1, c1, y, sc.hyp2f1(a1, b1, c1, y))
    return c * y1 * sc.hyp2f1(a1, b1, c1, y)


def sol(l, m, alpha, Q, H, TH, M, N):
    sum2 = 0.0
    for i in range(M + 1):
        sum1 = 0.0
        a = i * l
        for lst in multichoose(N, 2 * i):
            b = np.sum([i1 * i2 for i1, i2 in zip(list(range(1, N)), lst[1::])])
            sum1 += (Q ** i) * multinomial(lst) * (integral(a, b, m, 1.0) - integral(a, b, m, TH))
        sum2 += sum1
    Z = 1 - ((1 - m) / (alpha * m * H)) * sum2
    return Z


def find_TH0(TH, *data):
    l, m, alpha, Q, H, M, N = data
    return sol(l, m, alpha, Q, H, TH, M, N)

def find_THQ(TH, *data):
    l, m, Q = data
    return Q - Kr(l, m, TH)

# Q = 0.1
# data = (0.666, 0.512195121951220, Q)
# THini = 0.8
# THQ = fsolve(find_THQ, THini, args=data)
# print('THQ = ', THQ)
# data_B1 = (0.826, 0.453551912568306, 1.69, Q, 1, 10, 8)
#
# data = data_B1
# THini = 0.81
# TH0 = fsolve(find_TH0, THini, args=data)
# print('TH0 = ', TH0[0])


def integral2(a, b, l, m, x):
    a1 = - m * b
    b1 = m * (a * l + a + 1) - 1
    c1 = b1 + 1
    y = x ** (1 / m)
    y1 = x ** (b1 / m)
    c = m / b1
    F1 = sc.hyp2f1(a1 + m, b1, c1, y)
    F2 = sc.hyp2f1(a1 - m, b1, c1, y)
    F3 = sc.hyp2f1(a1, b1, c1, y)
    return c * y1 * (F1 + F2 - 2 * F3)

def sol2(l, m, alpha, Q, H, TH, M):
    sum2 = 0.0
    for i in range(M + 1):
        sum1 = 0.0
        for j in range(2 * i + 1):
            sum1 += (1 / (Q ** i)) * ((-1) ** j) * sc.binom(2 * i, j)  * (integral2(i, j, l, m, 1.0) - integral2(i, j, l, m, TH))
        sum2 += sum1
    Z = 1 + ((1 - m) / (alpha * m * H * Q)) * sum2
    return Z

## Subsoil 01
# "H" 1.0
# "m" 0.5885
# "l" 0.310
# "alpha" 5.51
# "Q" 0.01
# print(sc.hyp2f1(1, 2, 3, 0.5))

# Postprocessing
TH = np.linspace(0.6, 1, 501)
list_Z = []
for i in range(len(TH)):
    # Z = sol(0.310, 0.5885, 5.51, 0.01, 1, TH[i], 3, 25)
    # Z = sol(0.00055, 0.300699300699301, 2.4, 0.001, 2, TH[i], 1, 150)
    # Z = sol(0.054, 0.806201550387597, 1.4, 0.1, 1, TH[i], 6, 10)
    # Z = sol(0.0098, 0.393939393939394, 0.16, 0.01, 5, TH[i], 1, 50)
    # Z = sol(0.0065, 0.421965317919075, 1.57, 0.001, 1, TH[i], 1, 40)
    # Z = sol(0.054, 0.806201550387597, 1.4, -0.00001, 5, TH[i], 1, 107)
    # Z = sol(3.76, 0.774774774774775, 1.7, -3.870873400858568e-06, 2, TH[i], 1, 111)
    # Z = sol(0.054, 0.806201550387597, 1.4, -0.001, 1, TH[i], 1, 50)
    Z = sol(0.0065, 0.421965317919075, 1.57, -0.001, 1, TH[i], 1, 40)
    list_Z.append(Z)
    print(i, Z)

