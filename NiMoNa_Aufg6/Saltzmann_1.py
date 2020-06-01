import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d




h=0.01
T=600
p=0.1
q=1.5
r=1.1
x=np.zeros(int(T/h)+1)
y=np.zeros(int(T/h)+1)
z=np.zeros(int(T/h)+1)
t=np.zeros(int(T/h)+1)

x[0]=-np.sqrt(r-p)+0.0001
y[0]=np.sqrt(r-p)
z[0]=-np.sqrt(r-p)
t[0]=0


"""x=I=Eismasse
y=my=CO_2
z=theta=Ozeantemperatur
"""


def f_xpunkt(t,x,y,z,p,q,r):
    return -x-y

def g_ypunkt(t,x,y,z,p,q,r):
    return p*z+r*y-z**2*y

def h_zpunkt(t,x,y,z,p,q,r):
    return q*(x-z)


def Runge_Kutta(t,x,y,z,parameter,h):
    
    k_1_x=f_xpunkt(t,x,y,z,*parameter)
    l_1_y=g_ypunkt(t,x,y,z,*parameter)
    m_1_z=h_zpunkt(t,x,y,z,*parameter)
    
    k_2_x=f_xpunkt(t+h/2,x+h/2*k_1_x,y+h/2*l_1_y,z+h/2*m_1_z,*parameter)
    l_2_y=g_ypunkt(t+h/2,x+h/2*k_1_x,y+h/2*l_1_y,z+h/2*m_1_z,*parameter)
    m_2_z=h_zpunkt(t+h/2,x+h/2*k_1_x,y+h/2*l_1_y,z+h/2*m_1_z,*parameter)
    
    k_3_x=f_xpunkt(t+h/2,x+h/2*k_2_x,y+h/2*l_2_y,z+h/2*m_2_z,*parameter)
    l_3_y=g_ypunkt(t+h/2,x+h/2*k_2_x,y+h/2*l_2_y,z+h/2*m_2_z,*parameter)
    m_3_z=h_zpunkt(t+h/2,x+h/2*k_2_x,y+h/2*l_2_y,z+h/2*m_2_z,*parameter)
    
    k_4_x=f_xpunkt(t+h,x+h*k_3_x,y+h*l_3_y,z+h*m_3_z,*parameter)
    l_4_y=g_ypunkt(t+h,x+h*k_3_x,y+h*l_3_y,z+h*m_3_z,*parameter)
    m_4_z=h_zpunkt(t+h,x+h*k_3_x,y+h*l_3_y,z+h*m_3_z,*parameter)
    
    
    xneu=x+h/6*(k_1_x+2*k_2_x+2*k_3_x+k_4_x)
    yneu=y+h/6*(l_1_y+2*l_2_y+2*l_3_y+l_4_y)
    zneu=z+h/6*(m_1_z+2*m_2_z+2*m_3_z+m_4_z)
    tneu=t+h
    
    return tneu, xneu, yneu, zneu


for i in range(1,int(T/h)+1):
    
    t[i],x[i],y[i],z[i]=Runge_Kutta(t[i-1],x[i-1],y[i-1],z[i-1],[p,q,r],h)


plt.plot(t,x,"green",linewidth=1.5,label='Eismasse')
plt.plot(t,y,"orange",linewidth=1.5,label='CO_2')
plt.plot(t,z,"blue",linewidth=1.5,label='Ozeantemp.')
plt.legend(loc='upper right')
plt.xlabel('$t$ [a.u.]')
plt.ylabel('$y$')
plt.axis([80, 600, -5,5])


"""
ax=plt.axes(projection="3d")
ax.plot3D(x,y,z,'blue')
plt.show()
"""