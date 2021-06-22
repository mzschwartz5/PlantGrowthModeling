"""
Matthew Schwartz
One Dimensional Modeling of Plant Growth
Model 1

This program models plant growth in one dimension. It utilizes several different
numerical methods to solve a set of ordinary differential equations describing
plant growth. These methods include:
    - Euler Method
    - Adaptive Euler Method
    - 4th Order Runge-Kutta Method
    - 4th Order Adaptive Runge-Kutta Method

Equations to solve:
    - dL/dt = f(R)    where x = L(t), x is the end of the plant. 
    - h*(dR/dt) = g(R)*C - sigma*R
    - g(R) and f(R) are piecewise functions defined below.
    
Inputs:
    - Method:     ODE solver (suggestion: 3)
    - Tau:        Time step (suggestion: 0.0001)
    - nstep:      Number of steps (suggestion: 10000)
    - Anim:       Animate (suggestion: 2)

Outputs:
    - Animation: Growing plant
    - Graph: Growth Factor Concentration vs. Time
    - Graph: Plant Length vs. Time
    
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        g = 1.5*sigma*R
    elif Rg1 < R <= Rg2:
        g = 5*sigma*Rg1
    elif R > Rg2:
        g = 0
    else:
        print('Error: R must be greater than 0')
    
    return g

#%% Parameters and Initial Conditions

# Parameters
sigma = 0.009
Rf = 0.02
Rg1 = 0.01
Rg2 = 0.03
f0 = 3*Rf
h = 0.001
C = 1

err = 1.e-3
total_time = 100.

# User Inputs
method = int(input("Choose a number for a numerical method: \n\
 1-Euler\n 2-Adaptive Euler\n 3-4th Order Runge-Kutta\n 4-Adaptive R-K: "))
tau = float(input('Enter time step: '))
nstep = int(input('Enter number of steps: '))
anim = int(input('Animate? (1)Yes (2)No: '))

# Initial Conditions
Rplot = [Rf / 2.0]    # Growth factor starts below critical value
Lplot = [0]           # Plant length is 0 at t = 0
tplot = [0]

# Construct State Vector
state = np.array([Rplot[0],Lplot[0]])

#%% Numerical Method Functions

def euler(x,t,tau,derivsRK):
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

    F1 = derivsRK(x,t)
    xout = x + tau*F1
    return xout;

def eulera(x,t,tau,err,derivsRK):
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
      xTemp = euler(xSave,tSave,half_tau,derivsRK);
      t = tSave + half_tau;
      xSmall = euler(xTemp,t,half_tau,derivsRK);

      #* Take the single big time step
      t = tSave + tau;
      xBig = euler(xSave,tSave,tau,derivsRK);

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

def rk4(x,t,tau,derivsRK):
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
    F1 = derivsRK(x,t)
    t_half = t + half_tau
    xtemp = x + half_tau*F1
    F2 = derivsRK(xtemp,t_half)
    xtemp = x + half_tau*F2
    F3 = derivsRK(xtemp,t_half)
    t_full = t + tau
    xtemp = x + tau*F3
    F4 = derivsRK(xtemp,t_full)
    xout = x + tau/6.*(F1 + F4 + 2.*(F2+F3))
    return xout

def rka(x,t,tau,err,derivsRK):

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
        xTemp = rk4(xSave,tSave,half_tau,derivsRK)
        t = tSave + half_tau
        xSmall = rk4(xTemp,t,half_tau,derivsRK)
  
  #%* Take the single big time step
        t = tSave + tau
        xBig = rk4(xSave,tSave,tau,derivsRK)
  
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

def plantrk(s,t):
#%  Returns right-hand side of Plant ODE; used by Runge-Kutta routines
#%  Inputs
#%    s      State vector [R(1) L(1)]
#%    t      Time (not used)
#%  Output
#%    deriv  Derivatives [dR/dt(1) dLdt(1)]

    drdt = (C/h)*g(s[0]) - (sigma/h)*s[0]
    dldt = f(s[0])
    
#%* Return derivatives
    derivs = np.array([drdt, dldt])
    return derivs


#%% Solve ODE for R
time = 0.  # Initialize time
for istep in range(0,nstep):    ###  MAIN LOOP  ###

    if time > total_time:
        print('time completed in istep=',istep)
        break
    
    if ( method == 1 ):       # Euler Method
        state = euler(state,time,tau,plantrk)
        time = time + tau
        
    if ( method == 2 ):       # Adaptive Euler Method
        [state, time, tau] = eulera(state,time,tau,err,plantrk)
        
    if ( method == 3 ):       # 4th Order Runge-Kutta
        state = rk4(state,time,tau,plantrk)
        time = time + tau
    
    if ( method == 4 ):       # Adaptive 4th Order Runge-Kutta
        [state, time, tau] = rka(state,time,tau,err,plantrk)
    
    # Append to plot vectors
    Rplot.append(state[0])
    Lplot.append(state[1])
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

x = np.zeros(len(Lplot))

def update(num, x, y, line):
    line.set_data(x[:num], y[:num])
    line.axes.axis([-1,1,0,Lplot[-1]])
    return line,

# Animation
if (anim == 1):

    fig, ax = plt.subplots()
    line, = ax.plot(x,Lplot,color='k')
    
    ani = animation.FuncAnimation(fig, update, len(tplot), fargs=[x,Lplot,line],
                                  interval=25, blit=False)
    
    plt.show()
    plt.title('Plant Growth Animation')
    plt.xlabel('Length of Plant')
    
