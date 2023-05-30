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

def G1(a, b, n, x):
    m = 1 - 1/n
    a1 = m * (b - a)
    b1 = 1/n + m * b
    c1 = 1 + b1
    y = - x ** n
    p = 1 + n * m * b
    y1 = x ** p
    c = 1 / p
    return c * y1 * sc.hyp2f1(a1, b1, c1, y)

def G2(a, b, n, x):
    m = 1 - 1/n
    a1 = m * (b - a)
    b1 = 1/n + m * b
    c1 = 1 + b1
    y = - x ** n
    p = n * m * b
    y1 = x ** p
    return y1 * sc.hyp2f1(a1, b1, c1, y)

def G3(a, b, n, x):
    m = 1 - 1/n
    a1 = m * (b - a)
    b1 = 1/n + m * b
    c1 = 1 + b1
    y = - x ** n
    p = 1 + n * m * b
    y1 = x ** (n - 1 + p)
    c = n * a1 / (n + p)
    return c * y1 * sc.hyp2f1(1+a1, 1+b1, 1+c1, y)

def sol(Zb, Psib, n, l, Q, Psi, M, N):
    sum2 = 0.0
    for i in range(M + 1):
        sum1 = 0.0
        a = l * i
        for lst in multichoose(N, 2 * i):
            b = np.sum([i1 * i2 for i1, i2 in zip(list(range(1, N)), lst[1::])])
            sum1 += (Q ** i) * multinomial(lst) * (G1(a, b, n, Psi) - G1(a, b, n, Psib))
        sum2 += sum1
    Z = Zb - sum2
    return Z

def Ks(zb, psib, n, l, alpha, q, psi, z, N):
    Psib = - alpha * psib
    Psi = - alpha * psi
    sum1 = 0.0
    a = l
    for lst in multichoose(N, 2):
        b = np.sum([i1 * i2 for i1, i2 in zip(list(range(1, N)), lst[1::])])
        sum1 += multinomial(lst) * (G1(a, b, n, Psi) - G1(a, b, n, Psib))
    Ksat = (q / (alpha*(zb - z + psi))) * sum1
    return Ksat

def dpsidKs(zb, psib, n, l, alpha, q, psi, z, N):
    Psib = - alpha * psib
    Psi = - alpha * psi
    sum1 = 0.0
    a = l
    for lst in multichoose(N, 2):
        b = np.sum([i1 * i2 for i1, i2 in zip(list(range(1, N)), lst[1::])])
        sum1 += multinomial(lst) * (G2(a, b, n, Psi) - G3(a, b, n, Psib))
    dksdpsi = - (Ks(zb, psib, n, l, alpha, q, psi, z, N) + q * sum1) / (zb - z + psi - psib)
    return 1 / dksdpsi


def f(p, *data):
    N = 50
    ks, alpha, n, l = p
    z, psi, psib, q = data
    Psib = - alpha * psib
    Psi = - alpha * psi
    sum1 = 0.0
    a = l
    for lst in multichoose(N, 2):
        b = np.sum([i1 * i2 for i1, i2 in zip(list(range(1, N)), lst[1::])])
        sum1 += (q/ks) * multinomial(lst) * (G1(a, b, n, Psi) - G1(a, b, n, Psib))

    return z - psi + psib + sum1/alpha

def f3(p, *data):
    N = 50
    ks, alpha, n = p
    z, psi, psib, q, l = data
    Psib = - alpha * psib
    Psi = - alpha * psi
    sum1 = 0.0
    a = l
    for lst in multichoose(N, 2):
        b = np.sum([i1 * i2 for i1, i2 in zip(list(range(1, N)), lst[1::])])
        sum1 += (q/ks) * multinomial(lst) * (G1(a, b, n, Psi) - G1(a, b, n, Psib))

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

def equation_13(p, *data):
    zb, z, psi, psib, q, l = data
    data1 = (z, psi, psib, q, l)
    eq = zb - f3(p,*data1)
    return eq

def equations3(p, *data):
    z1, psi1, z2, psi2, psib, q, l = data
    data1 = (z1, psi1, psib, q, l)
    data2 = (z2, psi2, psib, q, l)
    eq = f3(p, *data1) - f3(p, *data2)
    return eq

def system_of_equations3(p, *data):
    zb, z1, psi1, z2, psi2, z3, psi3, psib, q, l = data
    data1 = (zb, z1, psi1, psib, q, l)
    data2 = (z1, psi1, z2, psi2, psib, q, l)
    data3 = (z2, psi2, z3, psi3, psib, q, l)
    eq1 = equation_13(p, *data1)
    eq2 = equations3(p, *data2)
    eq3 = equations3(p, *data3)
    return (eq1, eq2, eq3)

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


# ## S4
# Psi0 = 1.548120548322856
# psib = 0
# alpha = 0.0157
# n = 1.73
# ks = 4.48
# l = 0.0065
# Q = 0.001
# zb = 100
# M = 1
# N = 40


##S4
alpha = 0.0157
zb = 100.0
psib = 0
ks = 4.48
n = 1.73
l = 0.0065
Zb = alpha*zb
Psib = -alpha*psib
Q = 0.001
M = 1
N = 40

# ## S2
# Psi0 = 1.056786173108042
# psib = 0
# alpha = 0.014
# n = 5.16
# ks = 1.08
# l = 0.054
# Q = 0.1
# zb = 500
# M = 7
# N = 10

Psi1 = 0.2
Psi2 = 0.6
Psi3 = 1.0
Psi4 = 1.35

Z1 = sol(Zb, Psib, n, l, Q, Psi1, M, N)
Z2 = sol(Zb, Psib, n, l, Q, Psi2, M, N)
Z3 = sol(Zb, Psib, n, l, Q, Psi3, M, N)
Z4 = sol(Zb, Psib, n, l, Q, Psi4, M, N)

eps = 0.0e-8
Psi1 = Psi1 + eps*Psi1
Psi2 = Psi2 + eps*Psi2
Psi3 = Psi3 + eps*Psi3
Psi4 = Psi4 + eps*Psi4

psi1 = - Psi1/alpha
psi2 = - Psi2/alpha
psi3 = - Psi3/alpha
psi4 = - Psi4/alpha
z1 = Z1/alpha
z2 = Z2/alpha
z3 = Z3/alpha
z4 = Z4/alpha

# z1 = 25.000000000000021
# psi1 = -74.416207135367529
# z2 = 50.000000000022958
# psi2 = -49.794318572229308
# z3 = 75.000000000001705
# psi3 = -24.947219475573888

q = ks*Q
# data = (zb, z1, psi1, z2, psi2, z3, psi3, z4, psi4, psib, q)
# pini = (4.3, 0.013, 1.7, 0.0065)
# ks, alpha, n, l = fsolve(system_of_equations, pini, args=data)
data = (zb, z1, psi1, z2, psi2, z3, psi3, psib, q, l)
pini = (4.0, 0.013, 1.7)
ks, alpha, n = fsolve(system_of_equations3, pini, args=data)
print ('ks = ', ks)
print ('alpha = ', alpha)
print ('n = ', n)
# print ('l = ', l)


def find_TH0(TH, *data):
    l, m, alpha, Q, H, M, N = data
    return sol(l, m, alpha, Q, H, TH, M, N)



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
#     Z = sol(2.4, 0.0, 1.43, 0.00055, 0.001, Psi[i], 1, 100)
#     list_Z.append(Z)
#     print(i, Z)


# zmes = 2.3038219712147185/0.024
# psimes = - 0.096/0.024
# eps = psimes*1.0e-4
# ks_ana = Ks(2.4/0.024, 0.0, 1.43, 0.00055, 0.024, 0.001*11.4, psimes, zmes, 100)
# ks_ana_per = Ks(2.4/0.024, 0.0, 1.43, 0.00055, 0.024, 0.001*11.4, psimes + eps, zmes, 100)
# dpsidKs_ana = dpsidKs(2.4/0.024, 0.0, 1.43, 0.00055, 0.024, 0.001*11.4, psimes, zmes, 100)
# dpsidKs_per = eps / (ks_ana_per - ks_ana)
# print('ks_ana = ', ks_ana)
# print('dpsidKs_ana = ', dpsidKs_ana)
# print('dpsidKs_per = ', dpsidKs_per)

## S1
Psi0 = 2.270112245942883
psib = 0
alpha = 0.024
n = 1.43
ks = 11.4
l = 0.00055
Q = 0.001
zb = 100
M = 1
N = 100

# ## S2
# Psi0 = 1.056786173108042
# psib = 0
# alpha = 0.014
# n = 5.16
# ks = 1.08
# l = 0.054
# Q = -0.001
# zb = 500
# M = 1
# N = 50

# ## S3
# Psi0 = 0.762946478871909
# psib = 0
# alpha = 0.0016
# n = 1.65
# ks = 27.6
# l = 0.0098
# Q = 0.01
# zb = 100
# M = 1
# N = 50
#
# ## S4
# Psi0 = 1.548120548322856
# psib = 0
# alpha = 0.0157
# n = 1.73
# ks = 4.48
# l = 0.0065
# Q = 0.001
# zb = 100
# M = 1
# N = 40

# ## Clay
# Psi0 = 0.4
# psib = 0
# alpha = 0.008
# n = 1.09
# ks = 4.8
# l = 0.5
# Q = 0.01
# zb = 100
# M = 7
# N = 10

#########
Zb = alpha*zb
Psib = -alpha*psib
q = Q*ks

Psi = np.linspace(0.0, Psi0, 101)
list_Z = []
list_psi_ana = []
list_sens_ks_ana = []
list_sens_ks_per = []
for i in range(len(Psi)):
    Z = sol(Zb, Psib, n, l, Q, Psi[i], M, N)
    z = Z/alpha
    psi = - Psi[i]/alpha
    eps = psi*1.0e-3
    ks_ana = Ks(zb, psib, n, l, alpha, q, psi, z, N)
    ks_ana_per = Ks(zb, psib, n, l, alpha, q, psi+eps, z, N)
    dpsidKs_per = eps / (ks_ana_per - ks_ana)
    dpsidKs_per_norm = (ks / psi) * dpsidKs_per
    dpsidKs_ana = dpsidKs(zb, psib, n, l, alpha, q, psi, z, N)
    dpsidKs_ana_norm = (ks/psi)*dpsidKs_ana
    print(i, dpsidKs_ana_norm)
    list_psi_ana.append(psi)
    list_sens_ks_ana.append(dpsidKs_ana_norm)
    list_sens_ks_per.append(dpsidKs_per_norm)
    list_Z.append(Z)
    print(i, Z)

# z_mes = 50.000000000022958
# psi_mes = -49.794318572229308
# z_mes = 75.000000000001705
# psi_mes = -24.947219475573888
# z_mes = 25.000000000000021
# psi_mes = -74.416207135367529
#
# z_mes = z1
# psi_mes = psi1
#
# ks_ana = Ks(zb, psib, n, l, alpha, q, psi_mes, z_mes, N)
# print('z_mes = ', z_mes)
# print('psi_mes = ', psi_mes)
# print('ks_ana = ', ks_ana)