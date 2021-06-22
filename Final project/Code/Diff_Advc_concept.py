"""This is a proof-of-concept of the Crank-Nicolson solution to the 1D
Diffusion-Advection equation. 

Inputs:
    - N:       Number of grid points (suggestion: 100)
    - tau:     time step (suggestion: 0.0001)
    - nstep:   number of steps (suggestion: 1000)
    - anim:    Animation (suggestion: 1)

Outputs:
    - Plot: Numerical solution to D-A equation (Animated)
    
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

# Initiate a grid
N = int(input('Enter number of grid points: '))
L = .1                       # System size (length)
h = L/(N-1)                  # Grid spacing
x = np.arange(0,N)*h         # Coordinates of grid points

tau = float(input('Enter time step: '))
nStep = int(input('Enter number of steps: '))
anim = int(input('Animate? (1)Yes (2)No: '))

# Parameters
u = 1.        # Advection coefficient
d = 0.001      # Diffusion coefficient
a = (tau*u) / (4*h)
b = (tau*d) / (2*(h**2))

# Initial condition: delta function
C = np.zeros(N)

# Boundary Conditions: 
C[0] = 1

# Initiate two intermediate matrices used to solve problem
M1 = diags([-a-b, 1+2*b, a-b], [-1, 0, 1], shape=(N, N)).toarray()
M2 = diags([b+a, 1-2*b, b-a], [-1, 0, 1], shape=(N, N)).toarray()

# Final Matrix:
M = np.linalg.inv(M1).dot(M2)

C0 = np.copy(C)
plt.figure()
# Loop over desired number of steps.
for iStep in range(0,nStep):  ## MAIN LOOP ##
    
    plt.clf()

    # Update C
    C = M.dot(C)    # written this way because C is a row vector
    C[0] = 1
    C[-1] = C[-2]
    
    # Plot
    if (anim == 1):
        plt.grid()
        plt.plot(x,C0,'--')
        plt.plot(x,C)
        plt.axis([0,0.1,-.1,1.1])
        plt.title('Diffusion-Advection on Plant Model')
        plt.xlabel('Position')
        plt.ylabel('Amplitude')
        plt.draw()
        plt.pause(0.001)
        plt.show()

if (anim == 2):
    plt.grid()
    plt.plot(x,C0,'--')
    plt.plot(x,C)
    plt.axis([0,0.1,-.1,1.1])
    plt.title('Diffusion-Advection on a Delta-Function')
    plt.xlabel('Amplitude')
    plt.ylabel('Position')