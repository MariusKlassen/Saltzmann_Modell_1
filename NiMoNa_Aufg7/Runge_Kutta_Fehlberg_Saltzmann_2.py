import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# Butcher-Tableau RK_Fehl:

alpha = np.array([0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0])
beta = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0, 0.0],
        [1932.0/2197.0, (-7200.0)/2197.0, 7296.0/2197.0, 0.0, 0.0, 0.0],
        [439.0/216.0, -8.0, 3680.0/513.0, (-845.0)/4104.0, 0.0, 0.0],
        [(-8.0)/27.0, 2.0, (-3544.0)/2565.0, 1859.0/4104.0, (-11.0)/40.0, 0.0]])
c = np.array([25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, (-1.0)/5.0, 0.0]) # coefficients for 4th order method
c_star = np.array([16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, (-9.0)/50.0, 2.0/55.0]) # coefficients for 5th order method
cerr=c-c_star # build the difference of both c and c_star for the error estimation





def RK_Fehl(f, u_old, function_parameters, h,t):
    """
    Runge-Kutta-integrator for arbitrary right-hand-side
    :param f:                       right-hand-side to be integrated
    :param u_old:                   solution of the previous step
    :param function_parameters:     list of parameters of the function f
    :param h:                       step-size
    :return:                        solution of the next step
    """
    k1 = h * f( u_old, *function_parameters )
    k2 = h * f( u_old + beta[1,0] * k1, *function_parameters )
    k3 = h * f( u_old + beta[2,0] * k1 + beta[2,1] * k2, *function_parameters  )
    k4 = h * f( u_old + beta[3,0] * k1 + beta[3,1] * k2 + beta[3,2] * k3, *function_parameters  )
    k5 = h * f( u_old + beta[4,0] * k1 + beta[4,1] * k2 + beta[4,2] * k3 + beta[4,3] * k4, *function_parameters  )
    k6 = h * f( u_old + beta[5,0] * k1 + beta[5,1] * k2 + beta[5,2] * k3 + beta[5,3] * k4 + beta[5,4] * k5, *function_parameters )
            
    

    epsilon = abs( cerr[0] * k1 + cerr[2] * k3 + cerr[3] * k4 + cerr[4] * k5 + cerr[5] * k6 ) 

    if len( np.shape( epsilon ) ) > 0:
            epsilon = max( epsilon )

    if epsilon >= epsilon_tol:
        h=h*(epsilon_tol/epsilon)**0.2
        
    if epsilon < epsilon_tol:
       h=h*(epsilon_tol/epsilon)**0.25
        
    t=t+h
    
    return u_old + c_star[0]*k1+ c_star[1]*k2 + c_star[2]*k3+ c_star[3]*k4+ c_star[4]*k5+ c_star[5]*k6,t,h,epsilon

   


def Saltzmann(u_old,r,p,q):
    x,y,z=u_old
    x_p=-x-y
    y_p=p*z+r*y-y*z**2
    z_p=q*(x-z)
    return np.array([x_p,y_p,z_p])




# Numerical Parameters:
    
h = 0.025
T = 1500
ts = np.arange(0., T, h)
epsilon_tol=0.0000000000001
t=np.zeros(ts.size)

# Model Parameters:

r=1.4
p=0.1
q=1.5
x_0=-np.sqrt(r-p)+0.0001
y_0=np.sqrt(r-p)
z_0=-np.sqrt(r-p)
u = np.zeros((3, ts.size))
u[:, 0] = np.array([x_0, y_0, z_0])
i=0

for i in range(ts.size-1):
    u[:, i+1], t[i+1],h,epsilon= RK_Fehl(Saltzmann, u[:, i], [r, p, q], h,t[i])
    #plt.plot(t[i],epsilon,",r")
    



plt.plot(t, u[0], "green",linewidth=1.5,label='Eismasse')
plt.plot(t, u[1], "orange",linewidth=1.5,label='CO_2')
plt.plot(t, u[2], "blue",linewidth=1.5,label='Ozeantemp.')
plt.legend(loc='upper left')
plt.xlabel('$t$ [a.u.]')
plt.ylabel('$y$')
plt.axis([0, 400, -5,5])

"""
ax=plt.axes(projection="3d")
ax.plot3D(u[0],u[1],u[2],'blue')
plt.show()
"""