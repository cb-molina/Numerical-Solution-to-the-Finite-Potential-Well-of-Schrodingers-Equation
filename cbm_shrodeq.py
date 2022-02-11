"""
Christian B. Molina
Phy 104B - WQ 2021
version 3.0
"""

import numpy as np
from scipy import constants
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt

f= open("values.txt","w+")

# Constants and Variables ********************************************************************************************************
# Physical Constants 
mp = constants.value('proton mass')                     # kilograms
hbar = constants.value('reduced Planck constant')       # Joules * sec

# Given Constants
L = 7.0e-15                                             # fentometers
u0 = 1.6021773e-12                                      # (MeV * Joules/MeV) = Joules

# Generated Constants and variables 
approxzero = 1e-14
E_values = np.arange(approxzero, u0, 1e-16)
E_n = []
coeff_A = []
coeff_B = []
coeff_C = []
coeff_G = []
xI = np.linspace(0, L, 100)
xII = np.linspace(-L,0,100)
xIII = np.linspace(L,L*2,100)


# Functions ********************************************************************************************************
def k(E):
    return np.sqrt((2. * mp * E)/(np.power(hbar,2.)))

def alpha(E):
    return np.sqrt((2.*mp * (u0 - E))/(np.power(hbar,2.)))

def transEq(k,a):
    return (2/np.tan(k*L)) + (a / k) - (k / a)

# Wave Equation terms
def weqI(x,E): 
    return np.power(((alpha(E)/k(E))*np.sin(k(E)*x)) + np.cos(k(E)*x),2) # # add coefficients to this for when solved for C, square this also!

def weqII(x,E):
    return np.power(np.exp(alpha(E)*x),2)

def weqIII(x,E):
    return np.power(np.exp(-alpha(E)*x),2) # add coefficients to this for when solved for C, square this also! ((alpha(E)/k(E))*np.sin(k(E)*L) + np.cos(k(E)*L))*np.exp(alpha(E)*L)*

"""
    The integral of weqIII should be a constant: I believe alpha^2
"""


# Computation ********************************************************************************************************

# Energy Eigenvalues 
values = transEq(k(E_values), alpha(E_values))
n = 0
print("Length: ", len(E_values))
while n < len(values):
    f.write("n = %d, value = %f" % (n,values[n]))
    if (values[n] < ((approxzero)/u0)) & (values[n] >= 0):
        E_n.append(E_values[n])
        f.write("       ++++++++++++++++++++++++++++++")
        f.write("       ****************************\n")
        # if values[n] >= 0:
        #     E_n.append(E_values[n])
        #     f.write("       ****************************\n")
        # else:
        #     f.write("\n")
    else:
        f.write("\n")
    n += 1
E_roots = [E_n[0], E_n[-1]]

# Solving for coefficients C
print("\n\n****Beginning of Output****")
print("\n============================\n")
print("- Finding the Coeffiencents -\n")
for i in E_roots: 
    # if i == E_roots[0]:
    #     print("First E-value: ", i)
    # else:
    #     print("Second E-value: ", i)

    # Integrating the terms of the wave equation
    integrated_II = integrate.quad(lambda x: weqII(x,i), -approxzero*1e-6,-approxzero)
    # print("This is the integral of section II: ",integrated_II)

    integrated_I = integrate.quad(lambda x: weqI(x,i), 0,L) 
    # print("This is the integral of section I: ",integrated_I)

    integrated_III = integrate.quad(lambda x: weqIII(x,i), L,1e-12)
    # print("This is the integral of section III: ",integrated_III, "\n\n")

    coeff_C.append(1 / np.sqrt(integrated_I[0] + integrated_II[0] + integrated_III[0])) # constant_C = [c1, c2]

coeff_B = coeff_C
j = 0
while j < len(coeff_C):
    coeff_A.append((alpha(E_roots[j]) * coeff_C[j]) / k(E_roots[j]))
    # coeff_G.append(( np.exp(alpha(E_roots[j])*L) * ((alpha(E_roots[j])/k(E_roots[j])))*coeff_C[j]*np.sin(k(E_roots[j])*L) + coeff_C[j] * np.cos(k(E_roots[j]) * L)))
    coeff_G.append(np.exp(alpha(E_roots[j]) * L) * (coeff_A[j]*np.sin(k(E_roots[j]) * L) + coeff_B[j] * np.cos(k(E_roots[j]) * L)))
    j += 1 

print("Here are the A constants: ", coeff_A)
print("Here are the B constants: ", coeff_B)
print("Here are the C constants: ", coeff_C)
print("Here are the G constants: ", coeff_G)


def waveEquationI(n, x, E):
    return coeff_A[n]*np.sin(k(E[n])*x) + coeff_B[n]*np.cos(k(E[n])*x)

def waveEquationII(n, x, E):
    return coeff_C[n]*np.exp(alpha(E[n])*x)

def waveEquationIII(n, x, E):
    return coeff_G[n]*np.exp(-alpha(E[n])*x)



# Outputs and Plotting ********************************************************************************************************
print("\n============================\n")
print("- Physical Constants -")
print("Mass of Proton: ", mp)
print("Reduced Planck Constant: ", hbar)
print("\n- Given Constants -")
print("Depth of Box (L): ", L)
print("Max Energy: ", u0)
print("\n- Generated Constants -")
print("Approx. Zero: ", approxzero)
print("\n- Results -")
print("Energy Values: [", E_roots[0], ", ", E_roots[1], "]")
print("\n============================\n")
print("****end of output****")

plt.figure(1)
plt.title(r'Transcendental Equation Result vs. Energy', fontsize=12, font = "serif")
plt.plot(E_values, values, "b-")
plt.plot(E_values,np.zeros(len(E_values)), "k-")
plt.ylim(-50, 50)
plt.xlim(0, E_values[-1])
plt.grid(b='True')
plt.xlabel('Energy [J]', fontsize = 13, font = "serif")
plt.ylabel(r'$\cot(kL) + \frac{\alpha}{k} - \frac{k}{\alpha}$', fontsize = 13, font = "serif", rotation = 0, labelpad = 20)

plt.figure(2)
plt.title(r'Finite Potential Well Wavefunction', fontsize=12, font = "serif")
plt.grid(b='True')
plt.xlabel('Length [m]', fontsize = 13, font = "serif")
plt.ylabel('Energy\n[J]', rotation = 0, fontsize = 13, font = "serif")
# Section II
plt.plot(xII,waveEquationII(0,xII, E_roots),'r',label = r'$E=3.2839*10^{-13}$ J')
plt.plot(xII,waveEquationII(1,xII, E_roots),'g',label = r'$E=1.1953*10^{-12}$ J')
# Section I
plt.plot(xI,waveEquationI(0,xI, E_roots),'r')
plt.plot(xI,waveEquationI(1,xI, E_roots),'g')
# Section III
plt.plot(xIII,waveEquationIII(0,xIII, E_roots),'r')
plt.plot(xIII,waveEquationIII(1,xIII, E_roots),'g')

plt.legend()
plt.show()

f.close()
