"""
Matthew Schwartz
One Dimensional Modeling of Plant Growth
Model 2

This program models plant growth in one dimension. It utilizes several different
numerical methods to solve a set of ordinary differential equations describing
plant growth. These methods include:
    - Euler Method
    - Adaptive Euler Method
    - 4th Order Runge-Kutta Method
    - 4th Order Adaptive Runge-Kutta Method
    
It also solves the one-dimensional diffusion-advection equation, using the 
Crank Nicolson Method.
    
Equations to solve:
    - dL/dt = f(R)    where x = L(t), x is the end of the plant. 
    - h*(dR/dt) = g(R)*C - sigma*R
    - g(R) and f(R) are piecewise functions
    - dC/dt = -u*dC/dx + d*d^2C/dx^2, the diffusion-advection equation
    
Inputs:
    - Method:     ODE solver (suggestion: 3)
    - Tau:        Time step (suggestion: 0.0001)
    - nstep:      Number of steps (suggestion: 10000)
    - N:          Number of grid points (suggestion: 100)

Outputs:
    - Graph: Growth Factor Concentration vs. Time
    - Graph: Plant Length vs. Time
    - Graph: Metabolite concentration at end-point vs. time
    
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags


#%% Construct f(R) and g(R)

def f(R):
    """ f is a piecewise linear function """
    
    if 0 <= R <= Rf:
        f = 0
    elif R > Rf:
        f = f0
    else:
        print('Error: R must be greater than 0')
        
    return f

def g(R):
    """ g is a piecewise linear function """
    
    if 0 <= R <= Rg1:
        g = .015*sigma*R
    elif Rg1 < R <= Rg2:
        g = 0.06*Rg1
    elif R > Rg2:
        g = 0
    else:
        print('Error: R must be greater than 0')
    
    return g

#%% Parameters and Initial Conditions

# User Inputs
method = int(input("Choose a number for a numerical method: \n\
 1-Euler\n 2-Adaptive Euler\n 3-4th Order Runge-Kutta\n 4-Adaptive R-K: "))
tau = float(input('Enter time step: '))
nstep = int(input('Enter number of steps: '))
N = int(input('Enter initial number of grid points: '))
anim = int(input('Animate? (1)Yes (2)No: '))


# Parameters
sigma = 0.009           # Growth factor decay coefficient
w = 0.001               # Width of meristem
d = 0.001               # Diffusion coefficient (of metabolites in xylem fluid)
Rf = 0.005              # Critical values of R and G functions
Rg1 = 0.01                          # " "
Rg2 = 0.05                          # " "
f0 = 1                              # " "

err = 1.e-3             # Used by adaptive routines
total_time = 100.;

# Initial Conditions
Rplot = [Rf / 2.0]       # Growth factor starts below critical value
Lplot = [0.01]           # Plant length is 0 at t = 0
Cplot = [0]
tplot = [0]

# Initiate a grid (for concentration of metabolites)
h = Lplot[0]/(N-1)        # Spacing of grid
x = np.arange(0,N)*h      # Initial coordinates of grid points
C = np.zeros(N)           # Initiate C (concentration of metabolites)
C[0] = 1                  # Boundary Condition
b = (tau*d) / (2*(h**2))  # Final parameter (depends on h so must be defined here)

# Construct State Vector
state = np.array([Rplot[0],Lplot[0]])
#%% Numerical Method Functions

def euler(x,C,t,tau,derivsRK):
# %  Euler integrator (1st order)
# % Input arguments -
# %   x = current value of dependent variable
# %   t = independent variable (usually time)
# %   tau = step size (usually timestep)
# %   derivsRK = right hand side of the ODE; derivsRK is the
# %             name of the function which returns dx/dt
# %             Calling format derivsRK(x,t,param).
# %  Output arguments -
# %   xout = new value of x after a step of size tau

    F1 = derivsRK(x,C,t)
    xout = x + tau*F1
    return xout;

def eulera(x,C,t,tau,err,derivsRK):
# % Adaptive Euler routine
# % Inputs
# %   x          Current value of the dependent variable
# %   t          Independent variable (usually time)
# %   tau        Step size (usually time step)
# %   err        Desired fractional local truncation error
# %   derivsRK   Right hand side of the ODE; derivsRK is the
# %              name of the function which returns dx/dt
# %              Calling format derivsRK(x,t,param).
# % Outputs
# %   xSmall     New value of the dependent variable
# %   t          New value of the independent variable
# %   tau        Suggested step size for next call to rka

#* Set initial variables
    tSave = t;  xSave = x;    # Save initial values
    safe1 = 0.9;  safe2 = 4.;  # Safety factors

    #* Loop over maximum number of attempts to satisfy error bound
    maxTry = 500;
    for iTry in range(0,maxTry):

      #* Take the two small time steps
      half_tau = 0.5 * tau;
      xTemp = euler(xSave,C,tSave,half_tau,derivsRK);
      t = tSave + half_tau;
      xSmall = euler(xTemp,C,t,half_tau,derivsRK);

      #* Take the single big time step
      t = tSave + tau;
      xBig = euler(xSave,C,tSave,tau,derivsRK);

      #* Compute the estimated truncation error
      scale = err * (np.abs(xSmall) + np.abs(xBig))/2.;
      xDiff = xSmall - xBig;
      errorRatio = np.max( abs(xDiff)/(scale + np.spacing(1)) );

      #* Estimate new tau value (including safety factors)
      tau_old = tau;
      tau = safe1*tau_old*errorRatio**(-0.5);
      tau = np.max([tau,tau_old/safe2]);
      tau = np.min([tau,safe2*tau_old]);

      #* If error is acceptable, return computed values
      if (errorRatio < 1) :
        xSmall = 2*xSmall - xBig; # higher order correction
        return xSmall, t, tau

    #* Issue error message if error bound never satisfied
    BaseException('ERROR: Adaptive Euler routine failed');

def rk4(x,C,t,tau,derivsRK):
#%  Runge-Kutta integrator (4th order)
#% Input arguments -
#%   x = current value of dependent variable
#%   t = independent variable (usually time)
#%   tau = step size (usually timestep)
#%   derivsRK = right hand side of the ODE; derivsRK is the
#%             name of the function which returns dx/dt
#%             Calling format derivsRK(x,t).
#% Output arguments -
#%   xout = new value of x after a step of size tau
    half_tau = 0.5*tau
    F1 = derivsRK(x,C,t)
    t_half = t + half_tau
    xtemp = x + half_tau*F1
    F2 = derivsRK(xtemp,C,t_half)
    xtemp = x + half_tau*F2
    F3 = derivsRK(xtemp,C,t_half)
    t_full = t + tau
    xtemp = x + tau*F3
    F4 = derivsRK(xtemp,C,t_full)
    xout = x + tau/6.*(F1 + F4 + 2.*(F2+F3))
    return xout

def rka(x,C,t,tau,err,derivsRK):

#% Adaptive Runge-Kutta routine
#% Inputs
#%   x          Current value of the dependent variable
#%   t          Independent variable (usually time)
#%   tau        Step size (usually time step)
#%   err        Desired fractional local truncation error
#%   derivsRK   Right hand side of the ODE; derivsRK is the
#%              name of the function which returns dx/dt
#%              Calling format derivsRK(x,t).
#% Outputs
#%   xSmall     New value of the dependent variable
#%   t          New value of the independent variable
#%   tau        Suggested step size for next call to rka

#%* Set initial variables
    tSave = t;  xSave = x    # Save initial values
    safe1 = .9;  safe2 = 4.  # Safety factors
    eps = np.spacing(1) # smallest value

#%* Loop over maximum number of attempts to satisfy error bound
    maxTry = 100

    for iTry in range(1,maxTry):
	
#%* Take the two small time steps
        half_tau = 0.5 * tau
        xTemp = rk4(xSave,C,tSave,half_tau,derivsRK)
        t = tSave + half_tau
        xSmall = rk4(xTemp,C,t,half_tau,derivsRK)
  
  #%* Take the single big time step
        t = tSave + tau
        xBig = rk4(xSave,C,tSave,tau,derivsRK)
  
  #%* Compute the estimated truncation error
        scale = err * (np.abs(xSmall) + np.abs(xBig))/2.
        xDiff = xSmall - xBig
        errorRatio = np.max( [np.abs(xDiff)/(scale + eps)] )
        
        #print safe1,tau,errorRatio
  
  #%* Estimate news tau value (including safety factors)
        tau_old = tau

        tau = safe1*tau_old*errorRatio**(-0.20)
        tau = np.max([tau,tau_old/safe2])
        tau = np.min([tau,safe2*tau_old])
  
  #%* If error is acceptable, return computed values
        if errorRatio < 1 : 
          # xSmall = xSmall #% +  (xDiff)/15
          #   xSmall = (16.*xSmall - xBig)/15. # correction
            return xSmall, t, tau  

#%* Issue error message if error bound never satisfied
    print ('ERROR: Adaptive Runge-Kutta routine failed')
    return

def crank(s,C_in):
#%  Returns updated C vector (as part of state vector)
#%  Utilizes Crank-Nicolson method    
#%  Inputs
#%    s      State vector [R, L]
#%    C_in   Concentration vector
#%  Output
#%    C_out  Updated concentration vector    
    
    N = len(C_in)
    a = (tau*f(s[0])) / (4*h)
    
    # Initiate two intermediate matrices used to solve problem
    M1 = diags([-a-b, 1+2*b, a-b], [-1, 0, 1], shape=(N, N)).toarray()
    M2 = diags([b+a, 1-2*b, b-a], [-1, 0, 1], shape=(N, N)).toarray()

    # Final Matrix:
    M = np.linalg.inv(M1).dot(M2)

    C_out = C_in.dot(M)    # written this way because C is a row vector
    
    return C_out
    
def plantrk(s,C,t):
#%  Returns right-hand side of Plant ODE; used by Runge-Kutta routines
#%  Inputs
#%    s      State vector [R, L]
#%    C      Metabolite concentration vector
#%    t      Time (not used)
#%  Output
#%    deriv  Derivatives [dR/dt(1) dLdt(1)]

    drdt = ((C[-1])*g(s[0]) - sigma*s[0]) / w
    dldt = f(s[0])
    
#%* Return derivatives
    derivs = np.array([drdt, dldt])
    return derivs


#%% Solve ODE for R
time = 0.  # Initialize time
for istep in range(0,nstep):    ###  MAIN LOOP  ###
    
    # Record length before update
    L0 = np.copy(state[1])
    
    # Methods to solve for L(t)
    
    if ( method == 1 ):       # Euler Method
        state = euler(state,C,time,tau,plantrk)
        time = time + tau
        
    if ( method == 2 ):       # Adaptive Euler Method
        if time > total_time:
            print('time completed in istep=',istep)
            break
        [state, time, tau] = eulera(state,C,time,tau,err,plantrk)
        
    if ( method == 3 ):       # 4th Order Runge-Kutta
        state = rk4(state,C,time,tau,plantrk)
        time = time + tau
    
    if ( method == 4 ):       # Adaptive 4th Order Runge-Kutta
        if time > total_time:
            print('time completed in istep=',istep)
            break
        [state, time, tau] = rka(state,C,time,tau,err,plantrk)
    
    ## Solve for and update concentration C ##
    C = crank(state,C)
    
    # Lengthen C vector by change in L and use boundary conditions
    dL = state[1] - L0               # Change in L
    dN = np.int(np.round(dL / h))    # Number of new grid points
    
    # Apply BCs to fill in new grid points.
    C[0] = 1
    Ci = (1-(g(state[0])*(h/d)))*C[-1]
    if dL > 0:
        C = np.append(C,dN*[Ci])
    else:
        C[-1] = Ci

    # Print statement to denote step
    if ( istep % 1000 ) == 0:
        print('Number of iterations completed:',istep)
    
    # Append to plot vectors
    Rplot.append(state[0])
    Lplot.append(state[1])
    Cplot.append(C[-1])
    tplot.append(time)
    
        
#%% Plotting

# Non-animated plots of L(t) and R(t)
plt.figure()
plt.plot(tplot,Lplot)
plt.title('Length of Plant vs Time'); plt.xlabel('Time'); plt.ylabel('Length');
plt.grid()

plt.figure()
plt.plot(tplot,Rplot)
plt.title('Growth Factor Concentration vs. Time'); plt.xlabel('Time'); plt.ylabel('Concentration')
plt.grid()

plt.figure()
plt.plot(tplot,Cplot)
plt.title('Metabolite Concentration vs. Time'); plt.xlabel('Time'); plt.ylabel('Concentration')
plt.grid()
    
