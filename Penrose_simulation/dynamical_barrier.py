
# coding: utf-8

# In[76]:

import math
import numpy as np
from bqplot import pyplot as plt
from scipy.optimize import fsolve
from scipy.special import ellipeinc
#from tqdm import tqdm
#import geometry_3 as geo
#from geometry_3 import geometry
#import multiprocessing
#from multiprocessing import Pool
savepath=""

#fp2=open("Schoch_point_decay.txt","w")
# In[77]:
def plot_line(ipt, fpt, col):
    q = 10
    xstep=(fpt[0]-ipt[0])/q
    ystep=(fpt[1]-ipt[1])/q

    if(ipt[0]==fpt[0]):
        x = np.ones(q+1)
        x *= ipt[0]

        y=np.arange(ipt[1],fpt[1],ystep)
        y=np.append(y,[fpt[1]])
    if(ipt[1]==fpt[1]):
        x=np.arange(ipt[0],fpt[0],xstep)
        x=np.append(x,[fpt[0]])
        y = np.ones(q+1)
        y *= ipt[1]

    if(ipt[0]==fpt[0] and ipt[1]==fpt[1]):
        x = np.ones(q+1)
        x *= ipt[0]
        y = np.ones(q+1)
        y *= ipt[1]
    elif((ipt[0] != fpt[0]) and (ipt[1] != fpt[1])):
        x=np.arange(ipt[0],fpt[0],xstep)
        x=np.append(x,[fpt[0]])
        y=np.arange(ipt[1],fpt[1],ystep)
        y=np.append(y,[fpt[1]])


    if(col==0):
        plt.plot(x,y,'k')
    if(col==1):
        plt.plot(x,y,'b', linewidth = 0.5)
    if(col==2):
        plt.plot(x,y,'r', linewidth = 0.5)
    if(col==3):
        plt.plot(x,y,'m',linewidth=0.1)

# In[79]:

def plot_ellipse(a,b,xdis,ydis,iang,fang): #xdisplacement, ydisplacement, initial_angle, final_angle
    q=100
    step = (fang-iang)/q
    x=np.ones(q+1)
    y=np.ones(q+1)

    for t in range(q+1):
        theta = iang + step*t
        x[t] *= (xdis+a*np.cos(theta))
        y[t] *= (ydis+b*np.sin(theta))

    plt.plot(x,y,'k')

def vect_angle(u,v): #u, v are normalized vectors
    return math.acos(np.dot(u,v))


# In[82]:

def angle_from_0(v): #v is a normalized vector
    unit = np.array([1,0])
    if(np.array_equal(v, unit)):
        return 0
    if(np.array_equal(v, np.array([-1,0]))):
        return np.pi
    elif(v[1]>0):
        return vect_angle(unit, v)
    elif(v[1]<0):
        return 2*np.pi-vect_angle(unit,v)

def ell_func(t,a,b,D):
    return b*ellipeinc(t, (b**2-a**2)/b**2)-D


# In[104]:

#returns angle of the point with arc length D from the point of theta=0
def ellipt_int_angle(a, b, D):

    theta = fsolve(ell_func,x0=0, args=(a,b,D))

    return theta  #inverse(ellipt_int_length) should return angle
##print ellipt_int_angle(0.003,0.001945,0.00785)[0]*180/np.pi
# In[105]:

#returns the arc length from the point of theta=0 at phi
def ellipt_int_length(phi, a, b):

    return b*ellipeinc(phi, (b**2-a**2)/b**2)

##print ellipt_int_length(np.pi,0.003,0.001945)

def f2(kx):
    return np.arctan((-2*rho*ks**4*(2*kx**2-ks**2)**2*np.sqrt(kx**2-kp**2)*np.sqrt(kf**2-kx**2))/((2*kx**2-ks**2)**4*(kf**2-kx**2)+16*(kx**2-kp**2)*(ks**2-kx**2)*(kf**2-kx**2)*kx**4-rho**2*ks**8*(kx**2-kp**2)))


# In[100]:

def f3(kx):
    return np.arctan((-2*rho*ks**4*np.sqrt(kx**2-kp**2)*np.sqrt(kf**2-kx**2)*((2*kx**2-ks**2)**2-4*np.sqrt(kx**2-kp**2)*np.sqrt(kx**2-ks**2)*kx**2))/(((2*kx**2-ks**2)**2-4*np.sqrt(kx**2-kp**2)*np.sqrt(kx**2-ks**2)*kx**2)**2*(kf**2-kx**2)-rho**2*ks**8*(kx**2-kp**2)))


# In[101]:

def delta(theta_p):
    h = 10**(-10)
    kx = kf*np.sin(theta_p)
    D=0

    if(theta_p != np.pi/2):
        if(kx <= kp):
            D = 0
        elif(kp < kx and kx < ks):
            D = -(f2(kx+h)-f2(kx-h))/(2*h)
        elif(ks <= kx):
            D = -(f3(kx+h)-f3(kx-h))/(2*h)


    return D
# In[78]:
class geometry:
    def __init__(self,a,b,a1,b1,c,focdist,cut_angle):
        self.a=a
        self.b=b
        self.a1=a1
        self.b1=b1
        self.c=c    
        self.focdist=focdist
        self.cut_angle=cut_angle
    # In[80]:
    
#    def draw_walls():
#        a=0.003
#        b=0.0048
#        a1=0.003
#        b1=0.001945
#        c=0.0009
#    
#        k = np.sqrt(np.absolute(a**2-b**2))
#    
#    
#        w=0
#    
#        ipt = np.array([-a1,b])
#        fpt = np.array([-c,b])
#        plot_line(ipt,fpt,w)
#    
#        ipt = np.array([-a1,k])
#        fpt = np.array([-c,k])
#        plot_line(ipt,fpt,w)
#    
#        ipt = np.array([c,b])
#        fpt = np.array([a1,b])
#        plot_line(ipt,fpt,w)
#    
#        ipt = np.array([c,k])
#        fpt = np.array([a1,k])
#        plot_line(ipt,fpt,w)
#    
#        ipt = np.array([-a1,-k])
#        fpt = np.array([-c,-k])
#        plot_line(ipt,fpt,w)
#    
#        ipt = np.array([-a1,-b])
#        fpt = np.array([-c,-b])
#        plot_line(ipt,fpt,w)
#    
#        ipt = np.array([c,-k])
#        fpt = np.array([a1,-k])
#        plot_line(ipt,fpt,w)
#    
#        ipt = np.array([c,-b])
#        fpt = np.array([a1,-b])
#        plot_line(ipt,fpt,w)
#    
#        ipt = np.array([-c,k])
#        fpt = np.array([-c,b])
#        plot_line(ipt,fpt,w)
#    
#        ipt = np.array([-c,-k])
#        fpt = np.array([-c,-b])
#        plot_line(ipt,fpt,w)
#    
#        ipt = np.array([c,k])
#        fpt = np.array([c,b])
#        plot_line(ipt,fpt,w)
#    
#        ipt = np.array([c,-k])
#        fpt = np.array([c,-b])
#        plot_line(ipt,fpt,w)
#    
#        plot_ellipse(a,b,-a1,0,np.pi/2,3*np.pi/2)
#        plot_ellipse(a,b,a1,0,-np.pi/2,np.pi/2)
#        plot_ellipse(a1,b1,0,-k,0,np.pi)
#        plot_ellipse(a1,b1,0,k,np.pi,2*np.pi)
    
        #plt.show()

    # In[83]:
    
    #returns ([t, angle of normal vector on the intersection,curvenum])
    #returns ([t, angle of normal vector on the intersection,curvenum])
    def intersect_point(self,p, v):
        a=self.a
        b=self.b
        a1=self.a1
        b1=self.b1
        c=self.c
        focdist=self.focdist
        cut_angle=self.cut_angle
        k = np.sqrt(np.absolute(a**2-b**2))
    
    
        thresh = 10**(-10)
        tmin = 100.
        nangle = angle_from_0(v) #normal vector angle
        tangle = nangle+np.pi/2 #tangent vector angle
        curvenum = 0
        
        #1. y=b, -a1<=x<=-c
        #3. y=b, c<=x<=a1
        if(v[1] != 0):
            t = (b-p[1])/v[1]
            if(t>thresh):
                if((-a1<= p[0]+t*v[0]) and (p[0]+ t*v[0] <= -c)): #1
                    if(t < tmin):
                        curvenum = 1
                        tmin = t
                        nangle = 3*np.pi/2
                        ##print("\#1")
                if((c<= p[0]+t*v[0]) and (p[0]+ t*v[0] <= a1)): #3
                    if(t < tmin):
                        curvenum = 3
                        tmin = t
                        nangle = 3*np.pi/2
                        ##print("\#3")
    
        #2. y=k, -a1+dist<=x<=-c
        #4. y=k, c<=x<=a1-dist
        if(v[1]!=0):
            t = (k-p[1])/v[1]
            if(t>thresh):
                if((-a1+focdist<= p[0]+t*v[0]) and (p[0]+ t*v[0] <= -c)): #2
                    if(t < tmin):
                        curvenum = 2
                        tmin = t
                        nangle = np.pi/2
                        ##print("\#2")
                if((c<= p[0]+t*v[0]) and (p[0]+ t*v[0] <= a1-focdist)): #4
                    if(t < tmin):
                        curvenum = 4
                        tmin = t
                        nangle = np.pi/2
                        ##print("\#4")
    
        #5. y=-k, -a1+dist<=x<=-c
        #7. y=-k, c<=x<=a1-dist
        if(v[1]!=0):
            t = (-k-p[1])/v[1]
            if(t>thresh):
                if((-a1+focdist<= p[0]+t*v[0]) and (p[0]+ t*v[0] <= -c)): #5
                    if(t < tmin):
                        curvenum = 5
                        tmin = t
                        nangle = 3*np.pi/2
                        ##print("\#5")
                if((c<= p[0]+t*v[0]) and (p[0]+ t*v[0] <= a1-focdist)): #7
                    if(t < tmin):
                        curvenum = 7
                        tmin = t
                        nangle = 3*np.pi/2
                        ##print("\#7")
    
        #6. y=-b, -a1<=x<=-c
        #8. y=-b, c<=x<=a1
        if(v[1]!=0):
            t = (-b-p[1])/v[1]
            if(t>thresh):
                if((-a1<= p[0]+t*v[0]) and (p[0]+ t*v[0] <= -c)): #6
                    if(t < tmin):
                        curvenum = 6
                        tmin = t
                        nangle = np.pi/2
                        ##print("\#6")
                if((c<= p[0]+t*v[0]) and (p[0]+ t*v[0] <= a1)): #8
                    if(t < tmin):
                        curvenum = 8
                        tmin = t
                        nangle = np.pi/2
                        ##print("\#8")
    
        #9. x=-c, k<=y<=b
        #10. x=-c, -b<=y<=-k
        if(v[0]!=0):
            t = (-c-p[0])/v[0]
            if(t>thresh):
                if((k <= p[1]+t*v[1]) and (p[1]+ t*v[1] <= b)): #9
                    if(t < tmin):
                        curvenum = 9
                        tmin = t
                        nangle = np.pi
                        ##print("\#9")
                if((-b <= p[1]+t*v[1]) and (p[1]+ t*v[1] <= -k)): #10
                    if(t < tmin):
                        curvenum = 10
                        tmin = t
                        nangle = np.pi
                        ##print("\#10")
    
        #11. x=c, k<=y<=b
        #12. x=c, -b<=y<=-k
        if(v[0]!=0):
            t = (c-p[0])/v[0]
            if(t>thresh):
                if((k <= p[1]+t*v[1]) and (p[1]+ t*v[1] <= b)): #11
                    if(t < tmin):
                        curvenum = 11
                        tmin = t
                        nangle = 0
                        ##print("\#11")
                if((-b <= p[1]+t*v[1]) and (p[1]+ t*v[1] <= -k)): #12
                    if(t < tmin):
                        curvenum = 12
                        tmin = t
                        nangle = 0
                        ##print("\#12")
        #new lines
        #x=-a1+focdist, k-b1*np.sqrt(1-((a1-dist)/a1)**2)<=y<=k
        #x=-a1+focdist, -k<=y<=-k+b1*np.sqrt(1-((a1-dist)/a1)**2)
        #x=a1-dist, k-b1*np.sqrt(1-((a1-focdist)/a1)**2)<=y<=k
        #x=a1-dist, -k<=y<=-k+b1*np.sqrt(1-((a1-focdist)/a1)**2)
        r_cut=(-(a1-focdist)*np.cos(cut_angle)/a1**2+np.sqrt(((a1-focdist)*np.cos(cut_angle)/a1**2)**2-((a1-focdist)**2/a1**2-1)*(np.cos(cut_angle)**2/a1**2+np.sin(cut_angle)**2/b1**2)))/(np.cos(cut_angle)**2/a1**2+np.sin(cut_angle)**2/b1**2)
        if(v[1]!=np.tan(cut_angle)*v[0]):
            t = (np.tan(cut_angle)*(p[0]-(-a1+focdist))+(k-p[1]))/(v[1]-np.tan(cut_angle)*v[0])
            if(t>thresh):
                if((k-r_cut*np.sin(cut_angle) <= p[1]+t*v[1]) and (p[1]+ t*v[1] <= k)): #17
                    if(t < tmin):
                        curvenum = 17
                        tmin = t
                        nangle = np.pi/2+cut_angle
                
            t = (np.tan(np.pi-cut_angle)*(p[0]-(-a1+focdist))+(-k-p[1]))/(v[1]-np.tan(np.pi-cut_angle)*v[0])
            if(t>thresh):
                if((-k <= p[1]+t*v[1]) and (p[1]+ t*v[1] <= -k+r_cut*np.sin(cut_angle))): #18
                    if(t < tmin):
                        curvenum = 18
                        tmin = t
                        nangle = 3*np.pi/2-cut_angle
                        
            t = (np.tan(np.pi-cut_angle)*(p[0]-(a1-focdist))+(k-p[1]))/(v[1]-np.tan(np.pi-cut_angle)*v[0])
            if(t>thresh):
                if((k-r_cut*np.sin(cut_angle) <= p[1]+t*v[1]) and (p[1]+ t*v[1] <= k)): #19
                    if(t < tmin):
                        curvenum = 19
                        tmin = t
                        if(cut_angle<=np.pi/2):
                            nangle = np.pi/2-cut_angle
                        else:
                            nangle = 5*np.pi/2-cut_angle
            
            t = (np.tan(cut_angle)*(p[0]-(a1-focdist))+(-k-p[1]))/(v[1]-np.tan(cut_angle)*v[0])                        
            if(t>thresh):
                if((-k <= p[1]+t*v[1]) and (p[1]+ t*v[1] <= -k+r_cut*np.sin(cut_angle))): #20
                    if(t < tmin):
                        curvenum = 20
                        tmin = t
                        if(cut_angle<np.pi/2):
                            nangle = 3*np.pi/2+cut_angle
                        else:
                            nangle = cut_angle-np.pi/2
    
        #end
        #13.
        coeff = np.array([(v[0]/a)**2+(v[1]/b)**2, 2*v[0]*(p[0]+a1)/a**2 + 2*v[1]*p[1]/b**2, ((p[0]+a1)/a)**2+(p[1]/b)**2-1])
        root = np.roots(coeff)
        meet = 0
        if(root[0]>thresh and p[0]+root[0]*v[0]<-a1 and np.isreal(root[0])):
            t = root[0]
            meet = 1
        if(root[1]>thresh and p[0]+root[1]*v[0]<-a1 and np.isreal(root[1])):
            if(meet==0):
                t = root[1]
                meet = 1
            else:
                if(root[1]<root[0]):
                    t = root[1]
        if(meet==1 and t < tmin):
            tmin = t
            curvenum = 13
            ##print("\#13")
            x = p[0]+tmin*v[0]
            y = p[1]+tmin*v[1]
            if(y == 0):
                nangle = 0
            else:
                tangle = np.arctan(-((b/a)**2)*(x+a1)/y)
                if(tangle >= 0):
                    nangle = 3*np.pi/2 + tangle
                else:
                    nangle = np.pi/2 + tangle
    
    
        #14.
        coeff = np.array([(v[0]/a)**2+(v[1]/b)**2, 2*v[0]*(p[0]-a1)/a**2 + 2*v[1]*p[1]/b**2, ((p[0]-a1)/a)**2+(p[1]/b)**2-1])
        root = np.roots(coeff)
        meet = 0 #false
        if(root[0]>thresh and p[0]+root[0]*v[0]>a1 and np.isreal(root[0])):
            t = root[0]
            meet = 1 #true
        if(root[1]>thresh and p[0]+root[1]*v[0]>a1 and np.isreal(root[1])):
            if(meet==0):
                t = root[1]
                meet = 1
            else:
                if(root[1]<root[0]):
                    t = root[1]
            ##print("root[1]")
        if(meet==1 and t < tmin):
            ##print("\#14")
            curvenum = 14
            tmin = t
            x = p[0]+tmin*v[0]
            y = p[1]+tmin*v[1]
            if(y == 0):
                nangle = np.pi
            else:
                tangle = np.arctan(-((b/a)**2)*(x-a1)/y)
                if(tangle >= 0):
                    nangle = np.pi/2 + tangle
                else:
                    nangle = 3*np.pi/2 + tangle
    
    
        #15.
        coeff = np.array([(v[0]/a1)**2+(v[1]/b1)**2, 2*v[0]*(p[0])/a1**2 + 2*v[1]*(p[1]+k)/b1**2, ((p[0])/a1)**2+((p[1]+k)/b1)**2-1])
        root = np.roots(coeff)
        meet = 0
        if(root[0]!=root[1]):
            if(root[0]>thresh and p[1]+root[0]*v[1]>-k+r_cut*np.sin(cut_angle) and np.isreal(root[0])):
                t = root[0]
                meet = 1
            if(root[1]>thresh and p[1]+root[1]*v[1]>-k+r_cut*np.sin(cut_angle) and np.isreal(root[1])):
                if(meet==0):
                    t = root[1]
                    meet = 1
                else:
                    if(root[1]<root[0]):
                        t = root[1]
        if(meet==1 and t < tmin):
            ##print("\#15")
            curvenum = 15
            tmin = t
            x = p[0]+tmin*v[0]
            y = p[1]+tmin*v[1]
            tangle = np.arctan(-((b1/a1)**2)*(x/(y+k)))
            if(tangle == 0):
                nangle = np.pi/2
            else:
                nangle = np.pi/2 + tangle
    
        #16.
        coeff = np.array([(v[0]/a1)**2+(v[1]/b1)**2, 2*v[0]*(p[0])/(a1**2) + 2*v[1]*(p[1]-k)/(b1**2), ((p[0])/a1)**2+((p[1]-k)/b1)**2-1])
        root = np.roots(coeff)
        meet = 0
        if(root[0]!=root[1]):
            if(root[0]>thresh and p[1]+root[0]*v[1]<k-r_cut*np.sin(cut_angle) and np.isreal(root[0])):
                t = root[0]
                meet = 1
            if(root[1]>thresh and p[1]+root[1]*v[1]<k-r_cut*np.sin(cut_angle) and np.isreal(root[1])):
                if(meet==0):
                    t = root[1]
                    meet = 1
                else:
                    if(root[1]<root[0]):
                        t = root[1]
        if(meet==1 and t < tmin):
            ##print("\#16")
            curvenum = 16
            tmin = t
            x = p[0]+tmin*v[0]
            y = p[1]+tmin*v[1]
            tangle = np.arctan(-((b1/a1)**2)*(x/(y-k)))
            if(tangle == 0):
                nangle = 3*np.pi/2
            else:
                nangle = 3*np.pi/2 + tangle
    
        ##print("tmin = ",tmin)
        t_norm = np.array([tmin, nangle, curvenum])
        return t_norm
    
    
    # In[97]:
    
    def nangle_pt(self,iptx, ipty, curvenum):
        a=self.a
        b=self.b
        a1=self.a1
        b1=self.b1
        focdist=self.focdist
        cut_angle=self.cut_angle
        k = np.sqrt(np.absolute(a**2-b**2))
        nangle = np.pi/2
        if(curvenum == 1 or curvenum == 3 or curvenum == 5 or curvenum == 7):
            nangle = 3*np.pi/2
        elif(curvenum == 2 or curvenum == 4 or curvenum == 6 or curvenum == 8):
            nangle = np.pi/2
        elif(curvenum == 9 or curvenum == 10):
            nangle = np.pi
        elif(curvenum == 11 or curvenum == 12):
            nangle = 0
        elif(curvenum == 17):
            nangle = np.pi/2+cut_angle
        elif(curvenum == 18):
            nangle = 3*np.pi/2-cut_angle
        elif(curvenum == 19):
            if(cut_angle<=np.pi/2):
                nangle = np.pi/2-cut_angle
            else:
                nangle = 5*np.pi/2-cut_angle
        elif(curvenum == 20):
            if(cut_angle<np.pi/2):
                nangle = 3*np.pi/2+cut_angle
            else:
                nangle = cut_angle-np.pi/2
        elif(curvenum == 13):
            nangle=2*np.pi-np.arcsin((a*ipty/b)/(np.sqrt((a*ipty/b)**2+((b*(iptx+a1))/a)**2)))
        elif(curvenum == 14):
            nangle=np.pi+np.arcsin((a*ipty/b)/(np.sqrt((a*ipty/b)**2+((b*(iptx-a1))/a)**2)))
        elif(curvenum == 15):
            nangle=np.arccos((b1*iptx/a1)/(np.sqrt((a1*(ipty+k)/b1)**2+((b1*iptx)/a1)**2)))
        elif(curvenum == 16):
            nangle=2*np.pi-np.arccos((b1*iptx/a1)/(np.sqrt((a1*(ipty-k)/b1)**2+((b1*iptx)/a1)**2)))
    
        sc=np.array([np.cos(nangle),np.sin(nangle)])
        nangle=angle_from_0(sc)
        if(nangle>=2*np.pi):
            nangle=0
        return nangle
    
    
    # In[102]:
    def reflected_ray(self,iray): # iray = [initial point x, initial point y, angle]
        p = np.array([iray[0],iray[1]])
        v = np.array([np.cos(iray[2]),np.sin(iray[2])])
    
        t_norm = self.intersect_point(p,v)
        t = t_norm[0]
        norm = t_norm[1]
        curvenum = t_norm[2]
    
        x = p+t*v
    
        angle_diff = norm - angle_from_0((-1)*v)
        if(angle_diff>np.pi):
            angle_diff = angle_diff - 2*np.pi
        rangle = norm + angle_diff
        rangle = rangle % (2*np.pi)
    
    
        fray = np.array([x[0], x[1], rangle, curvenum, norm,t])
    
        return fray
    
    
    
    # * https://math.stackexchange.com/questions/433094/how-to-determine-the-arc-length-of-ellipse
    # * https://math.stackexchange.com/questions/436049/determining-the-angle-degree-of-an-arc-in-ellipse/436125#436125
    
    # In[106]:
    
    #for ellipses, given x and y points, determine the point that is in the distance D
    def shifted_pt_ell(self,iptx, ipty, curvenum, D, cw):
        if(curvenum == 13):
            theta = np.arcsin(ipty/b)
            theta = np.pi-theta
            iD = ellipt_int_length(theta,a,b)
    
            if(cw == 1):
                fD = iD - D
            if(cw == 0):
                fD = iD + D
    
            ftheta = ellipt_int_angle(a,b,fD)
    
            if(ftheta<np.pi/2):
                leftD = ellipt_int_length(np.pi/2,a,b) - fD
                res = np.array([-a1,b,leftD])
            if(ftheta>3*np.pi/2):
                leftD = fD - ellipt_int_length(3*np.pi/2,a,b)
                res = np.array([-a1,-b,leftD])
            if(np.pi/2<=ftheta and ftheta<=3*np.pi/2):
                fptx = -a1+a*np.cos(ftheta)
                fpty = b*np.sin(ftheta)
                res = np.array([fptx, fpty, 0])
    
    
        if(curvenum == 14):
            theta = np.arcsin(ipty/b)
            iD = ellipt_int_length(theta,a,b)
            if(cw == 1):
                fD = iD - D
            if(cw == 0):
                fD = iD + D
            ftheta = ellipt_int_angle(a,b,fD)
            if(ftheta<-np.pi/2):
                leftD = ellipt_int_length(-np.pi/2,a,b) - fD
                res = np.array([a1,-b,leftD])
            if(ftheta>np.pi/2):
                leftD = fD - ellipt_int_length(np.pi/2,a,b)
                res = np.array([a1,b,leftD])
            if(-np.pi/2<=ftheta and ftheta<=np.pi/2):
                fptx = a1+a*np.cos(ftheta)
                fpty = b*np.sin(ftheta)
                res = np.array([fptx, fpty, 0])
    
    
        if(curvenum == 15):
            theta = np.arccos(iptx/a1)
            iD = ellipt_int_length(theta,a1,b1)
    
            if(cw == 1):
                fD = iD + D
            if(cw == 0):
                fD = iD - D
    
            ftheta = ellipt_int_angle(a1,b1,fD)
    
            if(ftheta<0):
                leftD = -fD
                res = np.array([a1,-k,leftD])
            if(ftheta>np.pi):
                leftD = fD - ellipt_int_length(np.pi,a1,b1)
                res = np.array([-a1,-k,leftD])
            if(0<=ftheta and ftheta<=np.pi):
                fptx = a1*np.cos(ftheta)
                fpty = -k + b1*np.sin(ftheta)
                res = np.array([fptx, fpty, 0])
    
        if(curvenum == 16):
            theta = np.arccos(iptx/a1)
    
            iD = ellipt_int_length(theta,a1,b1)
    
            if(cw == 0):
                fD = iD + D
            if(cw == 1):
                fD = iD - D
    
            ftheta = ellipt_int_angle(a1,b1,fD)
    
            if(ftheta>np.pi):
                leftD = fD-ellipt_int_length(np.pi,a1,b1)
                res = np.array([-a1,k,leftD])
            if(ftheta<0):
                leftD = -fD
                res = np.array([a1,k,leftD])
            if(0<=ftheta and ftheta<=np.pi):
                fptx = a1*np.cos(ftheta)
                fpty = k - b1*np.sin(ftheta)
                res = np.array([fptx, fpty, 0])
    
    
        for i in range(2):
            res[i] = np.asscalar(res[i])
    
        return res
    
    
    # In[110]:
    
    def shifted_pt(iptx, ipty, rangle, curvenum, D, cw,count):
        fptx = iptx
        fpty = ipty
        sangle = rangle % (2*np.pi)
        count=count+1
        ###print cw
        #print ("-=======================================================-")
        #print ("D in this loop= %f") % D
        #print ("sangle=%f") % (sangle*180/np.pi)
        #print ("norm angle=%f") % (nangle_pt(iptx,ipty,curvenum)*180/np.pi)
        #print ("current position=(%f,%f)") % (fptx,fpty)
        #print ("function referred curvenum=%f") % curvenum
        #print ("current count=%d") % count
        if(curvenum == 1):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
    
            if(cw == 1):
                if(iptx+D<=-c):
                    fptx = iptx + D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("1 working")
                    return fpt
                else:
    
                    leftD = D - (-c-iptx)
                    norm_diff=-np.pi/2
                    curvenum = 9
                    #print ("position= (%f,%f)") % (fptx,fpty)
                    return shifted_pt(-c,ipty,rangle+norm_diff,curvenum,leftD,cw,count) # Basic of Recursion function
    
    
            else:
                if(-a1 <= iptx-D):
                    fptx = iptx - D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("1 working")
                    return fpt
                else:
                    leftD = D - (iptx-(-a1))
                    curvenum = 13
                    #print ("1 working")
                    return shifted_pt(-a1,ipty,rangle,13,leftD,cw,count)
    
    
        if(curvenum == 2):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
            if(cw == 0):
                if(iptx+D<=-c):
                    fptx = iptx + D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("2 working")
                    return fpt
                else:
    
                    leftD = D - (-c-iptx)
                    norm_diff=(np.pi-np.pi/2)
                    curvenum = 9
                    #print("2 working")
                    return shifted_pt(-c,ipty,rangle+norm_diff,9,leftD,cw,count)
    
    
            else:
                if(-a1 <= iptx-D):
                    fptx = iptx - D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("2 working")
                    return fpt
                else:
    
                    leftD = D - (iptx-(-a1))
                    norm_diff=(np.pi-nangle_pt(iptx,ipty,curvenum))
                    curvenum = 16
                    #print("2 working")
                    return shifted_pt(-a1,ipty,rangle+norm_diff,16,leftD,cw,count)
    
    
        if(curvenum == 3):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
            #if(rangle-nangle_pt(iptx,ipty,curvenum)>np.pi/2 and rangle-nangle_pt(iptx,ipty,curvenum)>np.pi/2<3*np.pi/2):
                ##print ("rangle-nagle_pt= %f>pi/2 and <3pi/2") % angle_diff
                ##print ("when cw=%f former one of curvenum %f is wrong") % (cw,curvenum)
    
            if(cw == 1):
                if(iptx+D<=a1):
                    fptx = iptx + D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("3 working")
                    return fpt
                else:
                    leftD = D - (a1-iptx)
                    curvenum = 14
                    #print("3 working")
                    return shifted_pt(a1,ipty,rangle,14,leftD,cw,count)
    
    
            else:
                if(c <= iptx-D):
                    fptx = iptx - D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("3 working")
                    return fpt
                else:
    
                    leftD = D - (iptx-c)
                    norm_diff=(2*np.pi-3*np.pi/2)
                    curvenum = 11
                    #print("3 working")
                    return shifted_pt(c,ipty,rangle+norm_diff,11,leftD,cw,count)
    
    
        if(curvenum == 4):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
            #if(rangle-nangle_pt(iptx,ipty,curvenum)>np.pi/2 and rangle-nangle_pt(iptx,ipty,curvenum)>np.pi/2<3*np.pi/2):
                ##print ("rangle-nagle_pt= %f>pi/2 and <3pi/2") % angle_diff
                ##print ("when cw=%f former one of curvenum %f is wrong") % (cw,curvenum)
    
            if(cw == 0):
                if(iptx+D<=a1):
                    fptx = iptx + D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("4 working")
                    return fpt
                else:
    
                    leftD = D - (a1-iptx)
                    norm_diff=(2*np.pi-np.pi/2)
                    curvenum = 16
                    #print("4 working")
                    return shifted_pt(a1,ipty,rangle+norm_diff,16,leftD,cw,count)
    
    
            else:
                if(c <= iptx-D):
                    fptx = iptx - D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("4 working")
                    return fpt
                else:
    
                    leftD = D - (iptx-c)
                    norm_diff=(0-np.pi/2)
                    curvenum = 11
                    #print("4 working")
                    return shifted_pt(c,ipty,rangle+norm_diff,11,leftD,cw,count)
    
    
        if(curvenum == 5):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
    
            if(cw == 1):
                if(iptx+D<=-c):
                    fptx = iptx + D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("5 working")
                    return fpt
                else:
    
                    leftD = D - (-c-iptx)
                    norm_diff=(np.pi-3*np.pi/2)
                    curvenum = 10
                    #print("5 working")
                    return shifted_pt(-c,ipty,rangle+norm_diff,10,leftD,cw,count)
    
            else:
                if(-a1 <= iptx-D):
                    fptx = iptx - D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("5 working")
                    return fpt
                else:
    
                    leftD = D - (iptx-(-a1))
                    norm_diff=(np.pi-3*np.pi/2)
                    curvenum = 15
                    #print("5 working")
                    return shifted_pt(-a1,ipty,rangle+norm_diff,15,leftD,cw,count)
    
    
        if(curvenum == 6):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
    
            if(cw == 0):
                if(iptx+D<=-c):
                    fptx = iptx + D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("6 working")
                    return fpt
                else:
    
                    leftD = D - (-c-iptx)
                    norm_diff=(np.pi-np.pi/2)
                    curvenum = 10
                    #print("6 working")
                    return shifted_pt(-c,ipty,rangle+norm_diff,10,leftD,cw,count)
    
    
            else:
                if(-a1 <= iptx-D):
                    fptx = iptx - D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("6 working")
                    return fpt
                else:
                    leftD = D - (iptx-(-a1))
                    curvenum = 13
                    #print("6 working")
                    return shifted_pt(-a1,ipty,rangle,13,leftD,cw,count)
    
    
        if(curvenum == 7):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
    
            if(cw == 1):
                if(iptx+D<=a1):
                    fptx = iptx + D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("7 working")
                    return fpt
                else:
                    leftD = D - (a1-iptx)
                    norm_diff=(0-3*np.pi/2)
                    curvenum = 15
                    #print("7 working")
                    return shifted_pt(a1,ipty,rangle+norm_diff,15,leftD,cw,count)
    
            else:
                if(c <= iptx-D):
                    fptx = iptx - D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("7 working")
                    return fpt
                else:
    
                    leftD = D - (iptx-c)
                    norm_diff=(2*np.pi-3*np.pi/2)
                    curvenum = 12
                    #print("7 working")
                    return shifted_pt(c,ipty,rangle+norm_diff,12,leftD,cw,count)
    
    
        if(curvenum == 8):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
            if(cw == 0):
                if(iptx+D<=a1):
                    fptx = iptx + D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("8 working")
                    return fpt
                else:
                    leftD = D - (a1-iptx)
                    curvenum = 14
                    #print("8 working")
                    return shifted_pt(a1,ipty,rangle,14,leftD,cw,count)
    
    
            else:
                if(c <= iptx-D):
    
                    fptx = iptx - D
                    fpty = ipty
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("8 working")
                    return fpt
                else:
    
                    leftD = D - (iptx-c)
                    norm_diff=(0-np.pi/2)
                    curvenum = 12
                    #print("8 working")
                    return shifted_pt(c,ipty,rangle+norm_diff,12,leftD,cw,count)
    
    
        if(curvenum == 9):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
            if(cw == 0):
                if(ipty+D<=b):
                    fptx = iptx
                    fpty = ipty + D
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("9 working")
                    return fpt
                else:
    
                    leftD = D - (b-ipty)
                    norm_diff=(3*np.pi/2-np.pi)
                    curvenum = 1
                    #print("9 working")
                    return shifted_pt(iptx,b,rangle+norm_diff,1,leftD,cw,count)
    
            else:
                if(k <= ipty-D):
                    fptx = iptx
                    fpty = ipty - D
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("9 working")
                    return fpt
                else:
    
                    leftD = D - (ipty-k)
                    norm_diff=(np.pi/2-np.pi)
                    curvenum = 2
                    #print("9 working")
                    return shifted_pt(iptx,k,rangle+norm_diff,2,leftD,cw,count)
    
    
        if(curvenum == 10):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
    
            if(cw == 0):
                if(ipty+D<=-k):
                    fptx = iptx
                    fpty = ipty + D
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("10 working")
                    return fpt
                else:
    
                    leftD = D - (-k-ipty)
                    norm_diff=(3*np.pi/2-np.pi)
                    curvenum = 5
                    #print("10 working")
                    return shifted_pt(iptx,-k,rangle+norm_diff,5,leftD,cw,count)
    
    
            else:
                if(-b <= ipty-D):
                    fptx = iptx
                    fpty = ipty - D
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("10 working")
                    return fpt
                else:
    
                    leftD = D - (ipty-(-b))
                    norm_diff=(np.pi/2-np.pi)
                    curvenum = 6
                    #print("10 working")
                    return shifted_pt(iptx,-b,rangle+norm_diff,6,leftD,cw,count)
    
    
        if(curvenum == 11):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
    
            if(cw == 1):
                if(ipty+D<=b):
                    fptx = iptx
                    fpty = ipty + D
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("11 working")
                    return fpt
                else:
    
                    leftD = D - (b-ipty)
                    norm_diff=(3*np.pi/2)
                    curvenum = 3
                    #print("11 working")
                    return shifted_pt(iptx,b,rangle+norm_diff,3,leftD,cw,count)
    
    
            else:
                if(k <= ipty-D):
                    fptx = iptx
                    fpty = ipty - D
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("11 working")
                    return fpt
                else:
    
                    leftD = D - (ipty-k)
                    norm_diff=(-3*np.pi/2)
                    curvenum = 4
                    #print("11 working")
                    return shifted_pt(iptx,k,rangle+norm_diff,4,leftD,cw,count)
    
    
    
        if(curvenum == 12):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
            if(cw == 1):
                if(ipty+D<=-k):
                    fptx = iptx
                    fpty = ipty + D
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("12 working")
                    return fpt
                else:
    
                    leftD = D - (-k-ipty)
                    norm_diff=3*np.pi/2
                    curvenum = 7
                    #print("12 working")
                    return shifted_pt(iptx,-k,rangle+norm_diff,7,leftD,cw,count)
    
    
            else:
                if(-b <= ipty-D):
                    fptx = iptx
                    fpty = ipty - D
                    fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                    #print("12 working")
                    return fpt
                else:
    
                    leftD = D - (ipty-(-b))
                    norm_diff=-3*np.pi/2
                    curvenum = 8
                    #print("12 working")
                    return shifted_pt(iptx,-b,rangle+norm_diff,8,leftD,cw,count)
    
    
        if(curvenum == 13):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
    
            res = shifted_pt_ell(iptx, ipty, 13, D, cw)
    
            if(res[2]==0):
                fptx = res[0]
                fpty = res[1]
                if(cw==1):
                    if(rangle<nangle_pt(iptx,ipty,13)-nangle_pt(fptx,fpty,13)):
                        norm_diff=(2*np.pi+nangle_pt(fptx,fpty,13)-nangle_pt(iptx,ipty,13))
                    else:
                        norm_diff=nangle_pt(fptx,fpty,13)-nangle_pt(iptx,ipty,13)
                    sangle=rangle+norm_diff
    
                else:
                    if(rangle+nangle_pt(fptx,fpty,13)-nangle_pt(iptx,ipty,13)>=2*np.pi):
                        norm_diff=nangle_pt(fptx,fpty,13)-nangle_pt(iptx,ipty,13)-2*np.pi
                    else:
                        norm_diff=(nangle_pt(fptx,fpty,13)-nangle_pt(iptx,ipty,13))
                    sangle=rangle+norm_diff
                fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                #print("13 working")
                return fpt
            else:
                if(cw==1):
                    curvenum = 1
                    if(rangle<nangle_pt(iptx,ipty,13)-3*np.pi/2):
                        norm_diff=(2*np.pi+3*np.pi/2-nangle_pt(iptx,ipty,13))
                    else:
                        norm_diff=3*np.pi/2-nangle_pt(iptx,ipty,13)
                    #print("13 working")
                    return shifted_pt(-a1, b, rangle+norm_diff, 1, res[2], cw,count)
    
    
                else:
                    curvenum = 6
                    if(rangle+np.pi/2-nangle_pt(iptx,ipty,13)>=2*np.pi):
                        norm_diff=(np.pi/2-nangle_pt(iptx,ipty,13))-2*np.pi
                    else:
                        norm_diff=(np.pi/2-nangle_pt(iptx,ipty,13))
                    #print("13 working")
                    return shifted_pt(-a1, -b, rangle+norm_diff, 6, res[2], cw,count)
    
    
        if(curvenum == 14):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
            res = shifted_pt_ell(iptx, ipty, 14, D, cw)
            if(res[2]==0):
                fptx = res[0]
                fpty = res[1]
    
                norm_diff=nangle_pt(fptx,fpty,14)-nangle_pt(iptx,ipty,14)
                sangle = rangle+norm_diff
                fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                #print("14 working")
                return fpt
    
            else:
                if(cw==1):
                    curvenum = 8
                    norm_diff=(np.pi/2-nangle_pt(iptx,ipty,14))
                    #print("14 working")
                    return shifted_pt(a1, -b, rangle+norm_diff, 8, res[2], cw,count)
    
    
    
                else:
                    curvenum = 3
                    norm_diff=(3*np.pi/2-nangle_pt(iptx,ipty,14))
                    #print("14 working")
                    return shifted_pt(a1, b, rangle+norm_diff, 3, res[2], cw,count)
    
    
    
        if(curvenum == 15):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
    
            res = shifted_pt_ell(iptx, ipty, 15, D, cw)
            if(res[2]==0):
                fptx = res[0]
                fpty = res[1]
    
                sangle = nangle_pt(fptx, fpty, curvenum) + angle_diff
                fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                #print("15 working")
                return fpt
            else:
                if(cw==1):
                    norm_diff=(3*np.pi/2-nangle_pt(iptx,ipty,15))
                    curvenum = 5
                    #print("15 working")
                    return shifted_pt(-a1, -k, rangle+norm_diff, 5, res[2], cw,count)
    
    
                else:
                    if(rangle+3*np.pi/2-nangle_pt(iptx,ipty,15)>=2*np.pi):
                        norm_diff=(3*np.pi/2-nangle_pt(iptx,ipty,15))-2*np.pi
                    else:
                        norm_diff=(3*np.pi/2-nangle_pt(iptx,ipty,15))
                    curvenum = 7
                    #print("15 working")
                    return shifted_pt(a1, -k, rangle+norm_diff, 7, res[2], cw,count)
    
    
    
    
        if(curvenum == 16):
            angle_diff = rangle - nangle_pt(iptx, ipty, curvenum)
    
    
            res = shifted_pt_ell(iptx, ipty, 16, D, cw)
            #print ("D=%f but res=shifted_pt_ell=%f") % (D,res[2])
            if(res[2]==0):
                fptx = res[0]
                fpty = res[1]
    
                sangle = nangle_pt(fptx, fpty, curvenum) + angle_diff
                fpt = np.array([fptx, fpty, sangle, curvenum, nangle_pt(fptx,fpty,curvenum),count])
                #print("16 working")
                return fpt
    
            else:
                if(cw==1):
                    if(rangle+(np.pi/2-nangle_pt(iptx,ipty,16))<0):
                        norm_diff=(5*np.pi/2-nangle_pt(iptx,ipty,16)) # Notice the angle 5*np.pi/2
                    else:
                        norm_diff=(np.pi/2-nangle_pt(iptx,ipty,16))
                    curvenum = 4
                    #print ()
                    #print("16 working")
                    return shifted_pt(a1, k, rangle+norm_diff, 4, res[2], cw,count)
    
    
    
                else:
                    norm_diff=(np.pi/2-nangle_pt(iptx,ipty,16))
                    curvenum = 2
                    #print("16 working")
                    return shifted_pt(-a1, k, rangle+norm_diff, 2, res[2], cw,count)
    
    
    
    
    
    # In[111]:
    
    #cw = isClockwise(fray[2], fray[4], D) #1 if cw, 0 if ccw
    def isClockwise(self,fangle, nangle,D):
        fvect = [np.cos(fangle),np.sin(fangle)]
        nvect = [np.cos(nangle), np.sin(nangle)]
        prod = np.cross(fvect, nvect)*D
    
        if(prod<0):
            cw = 1 #clockwise
        else:
            cw = 0 #counterclockwise
    
        return cw
    # In[140]:
    def eta(self,iptx,ipty,curvenum):
        a=self.a
        b=self.b
        a1=self.a1
        b1=self.b1
        c=self.c
        focdist=self.focdist
        cut_angle=self.cut_angle
        k = np.sqrt(np.absolute(a**2-b**2))
        length=0
        r_cut=(-(a1-focdist)*np.cos(cut_angle)/a1**2+np.sqrt(((a1-focdist)*np.cos(cut_angle)/a1**2)**2-((a1-focdist)**2/a1**2-1)*(np.cos(cut_angle)**2/a1**2+np.sin(cut_angle)**2/b1**2)))/(np.cos(cut_angle)**2/a1**2+np.sin(cut_angle)**2/b1**2)
        ellipse_phi=math.acos((a1-focdist+r_cut*np.cos(cut_angle))/a1)
        iDinit=ellipt_int_length(ellipse_phi,a1,b1)
        if(curvenum == 14):
            if(ipty>0):
                theta = np.arcsin(ipty/b)
                iD = ellipt_int_length(theta,a,b)
                length=iD
            else:
                theta = np.pi/2+np.arcsin(ipty/b)
                iD = ellipt_int_length(theta,a,b)
                length=self.eta(a,-b,8)+iD
        if(curvenum == 3):
            length=self.eta(a,b,14)+(a-iptx)
        if(curvenum == 11):
            length=self.eta(c,b,3)+(b-ipty)
        if(curvenum == 4):
            length=self.eta(c,k,11)+(iptx-c)
        if(curvenum == 19):
            length=self.eta(a1-focdist,k,4)+np.sqrt((iptx-(a1-focdist))**2+(k-ipty)**2)
        if(curvenum == 16):
            theta=np.arccos(iptx/a1)
            iD=ellipt_int_length(theta,a1,b1)
            length=self.eta(a1-focdist+r_cut*np.cos(cut_angle),k-r_cut*np.sin(cut_angle),19)+iD-iDinit
        if(curvenum == 17):
            length=self.eta(-a1+focdist-r_cut*np.cos(cut_angle),k-r_cut*np.sin(cut_angle),16)+np.sqrt((iptx-(-a1+focdist-r_cut*np.cos(cut_angle)))**2+(ipty-(k-r_cut*np.sin(cut_angle)))**2)
        if(curvenum == 2):
            length=self.eta(-a1+focdist,k,17)+(iptx+a1-focdist)
        if(curvenum == 9):
            length=self.eta(-c,k,2)+(ipty-k)
        if(curvenum == 1):
            length=self.eta(-c,b,9)+(-c-iptx)
        if(curvenum == 13):
            theta=np.pi/2-np.arcsin(ipty/b)
            iD=ellipt_int_length(theta,a,b)
            length=self.eta(-a1,b,1)+iD
        if(curvenum == 6):
            length=self.eta(-a1,-b,13)+(iptx+a1)
        if(curvenum == 10):
            length=self.eta(-c,-b,6)+(ipty+b)
        if(curvenum == 5):
            length=self.eta(-c,-k,10)+(-c-iptx)
        if(curvenum == 18):
            length=self.eta(-a1+focdist,-k,5)+np.sqrt((iptx-(-a1+focdist))**2+(ipty-(-k))**2)
        if(curvenum == 15):
            theta=np.pi-np.arccos(iptx/a1)
            iD=ellipt_int_length(theta,a1,b1)
            length=self.eta(-a1+focdist-r_cut*np.cos(cut_angle),-k+r_cut*np.sin(cut_angle),18)+iD-iDinit
        if(curvenum == 20):
            length=self.eta(a1-focdist+r_cut*np.cos(cut_angle),-k+r_cut*np.sin(cut_angle),15)+np.sqrt((iptx-(a1-focdist+r_cut*np.cos(cut_angle)))**2+(ipty-(-k+r_cut*np.sin(cut_angle)))**2)
        if(curvenum == 7):
            length=self.eta(a1-focdist,-k,20)+(a1-focdist-iptx)
        if(curvenum == 12):
            length=self.eta(c,-k,7)+(-k-ipty)
        if(curvenum == 8):
            length=self.eta(c,-b,12)+(iptx-c)
    
        return length
    # In[301]:
    
    def eta_inverse(self,D):
        a=self.a
        b=self.b
        a1=self.a1
        b1=self.b1
        c=self.c
        focdist=self.focdist
        cut_angle=self.cut_angle
        k = np.sqrt(np.absolute(a**2-b**2))
        r_cut=(-(a1-focdist)*np.cos(cut_angle)/a1**2+np.sqrt(((a1-focdist)*np.cos(cut_angle)/a1**2)**2-((a1-focdist)**2/a1**2-1)*(np.cos(cut_angle)**2/a1**2+np.sin(cut_angle)**2/b1**2)))/(np.cos(cut_angle)**2/a1**2+np.sin(cut_angle)**2/b1**2)
        ellipse_phi=math.acos((a1-focdist+r_cut*np.cos(cut_angle))/a1)
        Dinit=ellipt_int_length(ellipse_phi,a1,b1)
        curvenum=0
        if(D>=0 and D<self.eta(a1,b,3)):
            theta=ellipt_int_angle(a,b,D)[0]
            iptx=a*np.cos(theta)+a1
            ipty=b*np.sin(theta)
            curvenum=14
        elif(D>=self.eta(a1,b,3) and D<self.eta(c,b,11)):
            iptx=a1-(D-self.eta(a1,b,3))
            ipty=b
            curvenum=3
        elif(D>=self.eta(c,b,11) and D<self.eta(c,k,4)):
            iptx=c
            ipty=b-(D-self.eta(c,b,11))
            curvenum=11
        elif(D>=self.eta(c,k,4) and D<self.eta(a1-focdist,k,19)):
            iptx=c+(D-self.eta(c,k,4))
            ipty=k
            curvenum=4
        elif(D>=self.eta(a1-focdist,k,19) and D<self.eta(a1-focdist+r_cut*np.cos(cut_angle),k-r_cut*np.sin(cut_angle),16)):
            r_temp=D-self.eta(a1-focdist,k,19)
            iptx=a1-focdist+r_temp*np.cos(cut_angle)
            ipty=k-r_temp*np.sin(cut_angle)
            curvenum=19
        elif(D>=self.eta(a1-focdist+r_cut*np.cos(cut_angle),k-r_cut*np.sin(cut_angle),16) and D<self.eta(-a1+focdist-r_cut*np.cos(cut_angle),k-r_cut*np.sin(cut_angle),17)):
            theta=ellipt_int_angle(a1,b1,D-self.eta(a1-focdist+r_cut*np.cos(cut_angle),k-r_cut*np.sin(cut_angle),16)+Dinit)[0]
            iptx=a1*np.cos(theta)
            ipty=k-b1*np.sin(theta)
            curvenum=16
        elif(D>=self.eta(-a1+focdist-r_cut*np.cos(cut_angle),k-r_cut*np.sin(cut_angle),17) and D<self.eta(-a1+focdist,k,2)):
            r_temp=D-self.eta(-a1+focdist-r_cut*np.cos(cut_angle),k-r_cut*np.sin(cut_angle),17)
            iptx=-a1+focdist-(r_cut-r_temp)*np.cos(cut_angle)
            ipty=k-(r_cut-r_temp)*np.sin(cut_angle)
            curvenum=17
        elif(D>=self.eta(-a1+focdist,k,2) and D<self.eta(-c,k,9)):
            iptx=-a1+focdist+(D-self.eta(-a1+focdist,k,2))
            ipty=k
            curvenum=2
        elif(D>=self.eta(-c,k,9) and D<self.eta(-c,b,1)):
            iptx=-c
            ipty=k+(D-self.eta(-c,k,9))
            curvenum=9
        elif(D>=self.eta(-c,b,1) and D<self.eta(-a1,b,13)):
            iptx=-c-(D-self.eta(-c,b,1))
            ipty=b
            curvenum=1
        elif(D>=self.eta(-a1,b,13) and D<self.eta(-a1,-b,6)):
            theta=ellipt_int_angle(b,a,D-self.eta(-a1,b,13))[0]#Change Semimajor and Semiminor axis because it starts from end of b
            iptx=-a1-a*np.sin(theta)
            ipty=b*np.cos(theta)
            curvenum=13
        elif(D>=self.eta(-a1,-b,6) and D<self.eta(-c,-b,10)):
            iptx=-a1+(D-self.eta(-a1,-b,6))
            ipty=-b
            curvenum=6
        elif(D>=self.eta(-c,-b,10) and D<self.eta(-c,-k,5)):
            iptx=-c
            ipty=-b+(D-self.eta(-c,-b,10))
            curvenum=10
        elif(D>=self.eta(-c,-k,5) and D<self.eta(-a1+focdist,-k,18)):
            iptx=-c-(D-self.eta(-c,-k,5))
            ipty=-k
            curvenum=5
        elif(D>=self.eta(-a1+focdist,-k,18) and D<self.eta(-a1+focdist-r_cut*np.cos(cut_angle),-k+r_cut*np.sin(cut_angle),15)):
            r_temp=D-self.eta(-a1+focdist,-k,18)
            iptx=-a1+focdist-r_temp*np.cos(cut_angle)
            ipty=-k+r_temp*np.sin(cut_angle)
            curvenum=18
        elif(D>=self.eta(-a1+focdist-r_cut*np.cos(cut_angle),-k+r_cut*np.sin(cut_angle),15) and D<self.eta(a1-focdist+r_cut*np.cos(cut_angle),-k+r_cut*np.sin(cut_angle),20)):
            theta=ellipt_int_angle(a1,b1,D-self.eta(-a1+focdist-r_cut*np.cos(cut_angle),-k+r_cut*np.sin(cut_angle),15)+Dinit)[0]
            iptx=-a1*np.cos(theta)
            ipty=-k+b1*np.sin(theta)
            curvenum=15
        elif(D>=self.eta(a1-focdist+r_cut*np.cos(cut_angle),-k+r_cut*np.sin(cut_angle),20) and D<self.eta(a1-focdist,-k,7)):
            r_temp=D-self.eta(a1-focdist+r_cut*np.cos(cut_angle),-k+r_cut*np.sin(cut_angle),20)
            iptx=a1-focdist+(r_cut-r_temp)*np.cos(cut_angle)
            ipty=-k+(r_cut-r_temp)*np.sin(cut_angle)
            curvenum=20
        elif(D>=self.eta(a1-focdist,-k,7) and D<self.eta(c,-k,12)):
            iptx=a1-focdist-(D-self.eta(a1-focdist,-k,7))
            ipty=-k
            curvenum=7
        elif(D>=self.eta(c,-k,12) and D<self.eta(c,-b,8)):
            iptx=c
            ipty=-k-(D-self.eta(c,-k,12))
            curvenum=12
        elif(D>=self.eta(c,-b,8) and D<self.eta(a1,-b,14)):
            iptx=c+(D-self.eta(c,-b,8))
            ipty=-b
            curvenum=8
        elif(D>=self.eta(a1,-b,14) and D<self.eta(0,0,14)):
            theta=ellipt_int_angle(b,a,D-self.eta(a1,-b,14))[0]
            iptx=a1+a*np.sin(theta)
            ipty=-b*np.cos(theta)
            curvenum=14
        else:
            print("D is out of range")
        return np.array([iptx,ipty,curvenum])
    ##print eta_inverse(0.008)
    # In[301]:
    def Dynamical_Barrier(self,iptx,ipty,ang_diff,curvenum):
        a=self.a
        b=self.b
        a1=self.a1
        b1=self.b1
        c=self.c
        focdist=self.focdist
        cut_angle=self.cut_angle
        k = np.sqrt(np.absolute(a**2-b**2))
        r_cut=(-(a1-focdist)*np.cos(cut_angle)/a1**2+np.sqrt(((a1-focdist)*np.cos(cut_angle)/a1**2)**2-((a1-focdist)**2/a1**2-1)*(np.cos(cut_angle)**2/a1**2+np.sin(cut_angle)**2/b1**2)))/(np.cos(cut_angle)**2/a1**2+np.sin(cut_angle)**2/b1**2)
        ellipse_phi=math.acos((a1-focdist+r_cut*np.cos(cut_angle))/a1)
        xaxis=self.eta(iptx,ipty,curvenum)
        yaxis=np.sin(ang_diff)
        norm_vec=np.array([np.cos(self.nangle_pt(iptx,ipty,curvenum)),np.sin(self.nangle_pt(iptx,ipty,curvenum))])
        if((np.abs(ipty-np.tan(cut_angle)*iptx-(k-np.tan(cut_angle)*(-a1+focdist)))<1e-10) or (np.abs(ipty-np.tan(np.pi-cut_angle)*iptx-(-k-np.tan(np.pi-cut_angle)*(-a1+focdist)))<1e-10) or (np.abs(ipty-np.tan(np.pi-cut_angle)*iptx-(k-np.tan(np.pi-cut_angle)*(a1-focdist)))<1e-10) or (np.abs(ipty-np.tan(cut_angle)*iptx-(-k-np.tan(cut_angle)*(a1-focdist)))<1e-10)):
            return 4 #black zone
        else:
            if(xaxis>self.eta(-a1+focdist,k,2) and xaxis<self.eta(-a1+focdist,-k,18)):
                if(xaxis>self.eta(-a1,b,13) and xaxis<self.eta(-a1,-b,6)):
                    foc_vec=np.array([(-a1-iptx),(k-ipty)])
                    foc_vec=foc_vec/np.linalg.norm(foc_vec)
                    ##print ("normalized foc_vec=(%f,%f)") % (foc_vec[0],foc_vec[1])
                    crit_angle=vect_angle(foc_vec,norm_vec)
                    if(np.absolute(yaxis)>np.absolute(np.sin(crit_angle))):
                        return 1 #red zone
                    else:
                        return 3 #green zone
                else:
                    return 1 #red zone
            elif((xaxis<self.eta(a1-focdist,k,19)) or (xaxis>self.eta(a1-focdist,-k,7))):
                if(xaxis<self.eta(a1,b,3) or xaxis>self.eta(a1,-b,14)):
                    foc_vec=np.array([(a1-iptx),(k-ipty)])
                    foc_vec=foc_vec/np.linalg.norm(foc_vec)
                    crit_angle=vect_angle(foc_vec,norm_vec)
                    if(np.absolute(yaxis)>np.absolute(np.sin(crit_angle))):
                        return 2 #blue zone
                    else:
                        return 3 #green zone
                else:
                    return 2
            else:
                return 3
    
    ##print Dynamical_Barrier(-0.006,0.00,np.arcsin(np.pi/6),13)
    # In[300]:
    def IC_generator(self,norm_D,y,h1,h2,step,step2): #Set Initial Condition
        #a=0.003
        #b=0.0048
        #a1=0.003
        #b1=0.001945
        #c=0.0009
        #k = np.sqrt(np.absolute(a**2-b**2))
        initial=[]
        ray=np.array([0.,0.,0.,0.])
        #D=norm_D*eta(0,0,14,focdist)
        ##print ("D=%f") % D
        for i in range(step):
            D=(norm_D+h1*i/step)*self.eta(0,0,14)
            ##print D
            ray[0]=self.eta_inverse(D)[0]
            ##print eta_inverse(D)[0]
            ray[1]=self.eta_inverse(D)[1]
            ray[3]=self.eta_inverse(D)[2]
            for j in range(step2):
                ray[2]=self.nangle_pt(ray[0],ray[1],ray[3])-np.arcsin(y+h2*j/step2)
                ##print ray
                initial.append(np.array([ray[0],ray[1],ray[2],ray[3]]))    #Call by value
        return initial