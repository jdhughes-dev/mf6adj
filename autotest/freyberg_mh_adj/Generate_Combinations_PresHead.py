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


def integral(a, b, n, x, model):
    if model == 'VG':
        m = 1 - 1/n
    elif model == 'B':
        m = 1 - 2/n

    a1 = m * (b - a)
    b1 = 1/n + m * b
    c1 = 1 + b1
    y = - x ** n
    p = 1 + n * m * b
    y1 = x ** p
    c = 1 / p
    return c * y1 * sc.hyp2f1(a1, b1, c1, y)


def sol(Zb, Psib, n, gam, Q, Psi, M, N, model):
    if model == 'VG':
        beta = 2
    elif model == 'B':
        beta = 1
    sum2 = 0.0
    for i in range(M + 1):
        sum1 = 0.0
        a = - gam * i
        for lst in multichoose(N, beta * i):
            b = np.sum([i1 * i2 for i1, i2 in zip(list(range(1, N)), lst[1::])])
            sum1 += (Q ** i) * multinomial(lst) * (integral(a, b, n, Psi, model) - integral(a, b, n, Psib, model))
        sum2 += sum1
    Z = Zb - sum2
    return Z

def Ks(B, psib, n, l, alpha, q, psi, z, N, model):
    if model == 'VG':
        beta = 2
    elif model == 'B':
        beta = 1
    Psib = - alpha * psib
    Psi = - alpha * psi
    sum1 = 0.0
    a = - l
    for lst in multichoose(N, beta):
        b = np.sum([i1 * i2 for i1, i2 in zip(list(range(1, N)), lst[1::])])
        sum1 += multinomial(lst) * (integral(a, b, n, Psi, model) - integral(a, b, n, Psib, model))
    Ksat = (q / (alpha*(B - z + psi))) * sum1
    return Ksat


def f(p, *data):
    N = 50
    ks, alpha, n, l = p
    z, psi, psib, q = data
    model = 'VG'
    beta = 2
    Psib = - alpha * psib
    Psi = - alpha * psi
    sum1 = 0.0
    a = - l
    for lst in multichoose(N, beta):
        b = np.sum([i1 * i2 for i1, i2 in zip(list(range(1, N)), lst[1::])])
        sum1 += (q/ks) * multinomial(lst) * (integral(a, b, n, Psi, model) - integral(a, b, n, Psib, model))

    return z - psi + psib + sum1/alpha

def equation_1(p, *data):
    zb, z, psi, psib, q = data
    data1 = (z, psi, psib, q)
    eq = zb - f(p, *data1)
    return eq

def equations(p, *data):
    z1, psi1, z2, psi2, psib, q = data
    data1 = (z1, psi1, psib, q)
    data2 = (z2, psi2, psib, q)
    eq = f(p, *data1) - f(p, *data2)
    return eq

def system_of_equations(p, *data):
    zb, z1, psi1, z2, psi2, z3, psi3, z4, psi4, psib, q = data
    data1 = (zb, z1, psi1, psib, q)
    data2 = (z1, psi1, z2, psi2, psib, q)
    data3 = (z2, psi2, z3, psi3, psib, q)
    data4 = (z3, psi3, z4, psi4, psib, q)
    eq1 = equation_1(p, *data1)
    eq2 = equations(p, *data2)
    eq3 = equations(p, *data3)
    eq4 = equations(p, *data4)
    return (eq1, eq2, eq3, eq4)


a = 0.014
zb = 100.0
Psi1 = 0.2
Psi2 = 0.6
Psi3 = 1.0
Psi4 = 1.35
print('Psi2=',Psi2)
Z1 = sol(1.4, 0.0, 5.16, 0.054, 0.001, Psi1, 1, 50, 'VG')
Z2 = sol(1.4, 0.0, 5.16, 0.054, 0.001, Psi2, 1, 50, 'VG')
Z3 = sol(1.4, 0.0, 5.16, 0.054, 0.001, Psi3, 1, 50, 'VG')
Z4 = sol(1.4, 0.0, 5.16, 0.054, 0.001, Psi4, 1, 50, 'VG')
Psi1 = Psi1 + Psi1/100000
Psi2 = Psi2 + Psi2/100000
Psi3 = Psi3 + Psi3/100000
Psi4 = Psi4 + Psi4/100000
print('Psi2=',Psi2)

psi1 = - Psi1/a
psi2 = - Psi2/a
psi3 = - Psi3/a
psi4 = - Psi4/a
z1 = Z1/a
z2 = Z2/a
z3 = Z3/a
z4 = Z4/a
print('psi4=',psi4)
print('zb=',zb)
psib = 0.0
q = 1.08*0.001
data = (zb, z1, psi1, z2, psi2, z3, psi3, z4, psi4, psib, q)
pini = (1.07, 0.013, 5.5, 0.052)
ks, alpha, n, l = fsolve(system_of_equations, pini, args=data)
print ('ks = ', ks)
print ('alpha = ', alpha)
print ('n = ', n)
print ('l = ', l)

# def fn(p, *data):
#     N = 50
#     n = p
#     ks, alpha, l, z, psi, psib, q = data
#     model = 'VG'
#     beta = 2
#     Psib = - alpha * psib
#     Psi = - alpha * psi
#     sum1 = 0.0
#     a = - l
#     for lst in multichoose(N, beta):
#         b = np.sum([i1 * i2 for i1, i2 in zip(list(range(1, N)), lst[1::])])
#         sum1 += (q/ks) * multinomial(lst) * (integral(a, b, n, Psi, model) - integral(a, b, n, Psib, model))
#
#     return z - psi + psib + sum1/alpha
#
# def equation_1n(p, *data):
#     ks, alpha, l, zb, z, psi, psib, q = data
#     data1 = (ks, alpha, l, z, psi, psib, q)
#     eq = zb - fn(p, *data1)
#     return eq
#
# a = 0.014
# ks = 1.08
# l = 0.054
# zb = 100.0
# Psi1 = 0.5
# Z1 = sol(1.4, 0.0, 5.16, 0.054, 0.001, Psi1, 1, 50, 'VG')
# psi1 = - Psi1/a
# z1 = Z1/a
# psib = 0.0
# q = 1.08*0.001
# data = (ks, a, l, zb, z1, psi1, psib, q)
# pini = (4.2)
# n = fsolve(equation_1n, pini, args=data)
# print ('n = ', n)


# ks, alpha, n, l =  fsolve(equations, (1, 1))
#
# print equations((x, y))

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
Psi = np.linspace(0.0, 2.4, 101)
list_Z = []
# for i in range(len(Psi)):
#     # Z = sol(0.310, 0.5885, 5.51, 0.01, 1, TH[i], 3, 25)
#     # Z = sol(0.00055, 0.300699300699301, 2.4, 0.001, 2, TH[i], 1, 150)
#     # Z = sol(0.054, 0.806201550387597, 1.4, 0.1, 1, TH[i], 6, 10)
#     # Z = sol(0.0098, 0.393939393939394, 0.16, 0.01, 5, TH[i], 1, 50)
#     # Z = sol(0.0065, 0.421965317919075, 1.57, 0.001, 1, TH[i], 1, 40)
#     # Z = sol(0.054, 0.806201550387597, 1.4, -0.00001, 5, TH[i], 1, 107)
#     # Z = sol(1.4, 0.0, 5.16, 0.054, 0.1, Psi[i], 5, 15, 'VG')
#     # if Psi[i] < 0.6:
#     #     Z = sol(2.0, 0.0, 5.16, 0.054, 0.1, Psi[i], 3, 15, 'VG')
#     # else:
#     # Z = sol(2.0, 0.0, 5.16, 0.054, 0.1, Psi[i], 7, 12, 'VG')
#     Z = sol(1.4, 0.0, 5.16, 0.054, -0.005, Psi[i], 1, 70, 'VG')
#     Z = sol(1.4, 0.0, 5.16, 0.054, -0.001, Psi[i], 1, 50, 'VG')
#     Z = sol(2.4, 0.0, 1.43, 0.00055, 0.001, Psi[i], 1, 100, 'VG')
#     list_Z.append(Z)
#     print(i, Z)

# Ks(B, psib, n, l, alpha, q, psi, z, N, model)
# print(Ks(1.4/0.014, 0.0, 5.16, 0.054, 0.014, -0.001, -0.0149/0.014, 1.385114900000145/0.014, 50, 'VG'))
print(Ks(2.4/0.024, 0.0, 1.43, 0.00055, 0.024, 0.001*11.4, -2.2560000000000002/0.024, 0.02895998389036203/0.024, 100, 'VG'))