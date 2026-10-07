import sys, getopt
import os
import numpy as np
import h5py
from scipy import integrate
import math

dt2 = 3.51564
delta_output = 300
dt = dt2/delta_output
resolution_theta = 321
resolution_phi = 161
times = xrange(0,743400,delta_output)    

def readfield(folder, field):
    #Readig the variable to disk
    f= h5py.File(folder + "/" + field + '.hdf5', "r")
    ds = f[field]
    data = np.empty(shape=ds.shape, dtype=ds.dtype)
    data[:] = ds[:]
    f.close()
    return data
    
def surface_integral2(samples, h_theta, h_phi):
    integral = 0.0
  
    iphi = 0
    itheta = 0
    phi = iphi*h_phi
    theta = itheta*h_theta
    integral = integral + np.sin(phi)*samples[itheta, iphi]
     
    iphi = samples.shape[1] 
    itheta = samples.shape[0] 
    phi = iphi*h_phi
    theta = itheta*h_theta 
    integral = integral + np.sin(phi)*samples[itheta-1, iphi-1]

    iphi = 0
    itheta = samples.shape[0] 
    phi = iphi*h_phi
    theta = itheta*h_theta
    integral = integral + np.sin(phi)*samples[itheta-1, iphi]

    iphi = samples.shape[1] 
    itheta = 0
    phi = iphi*h_phi
    theta = itheta*h_theta
    integral = integral + np.sin(phi)*samples[itheta, iphi-1]
    
    for iphi in range(1, resolution_phi - 1):
        phi = iphi*h_phi
        for itheta in range(1, resolution_theta -1):
            theta = itheta*h_theta 
            integral = integral + 4.0*np.sin(phi)*samples[itheta, iphi]


    iphi = 0
    phi = iphi*h_phi
    for itheta in range(0, resolution_theta -1):
        theta = itheta*h_theta 
        integral = integral + 2.0*np.sin(phi)*samples[itheta, iphi]

    iphi = samples.shape[1] 
    phi = iphi*h_phi
    for itheta in range(0, resolution_theta -1):
        theta = itheta*h_theta 
        integral = integral + 2.0*np.sin(phi)*samples[itheta, iphi-1]

    itheta = 0
    theta = itheta*h_theta  
    for iphi in range(0, resolution_phi - 1):
        phi = iphi*h_phi
        integral = integral + 2.0*np.sin(phi)*samples[itheta, iphi]       

    itheta = samples.shape[0] 
    theta = itheta*h_theta    
    for iphi in range(0, resolution_phi - 1):
        phi = iphi*h_phi
        integral = integral + 2.0*np.sin(phi)*samples[itheta-1, iphi]           

    return 0.25*h_theta*h_phi*integral
      
def calculate(sphere, radius):
    
    f90 = open(sphere + '/s0_c00.dat', 'w')

    f100 = open(sphere + '/s0_c1m1.dat', 'w')
    f101 = open(sphere + '/s0_c10.dat', 'w')
    f102 = open(sphere + '/s0_c1p1.dat', 'w')

    f110 = open(sphere + '/s0_c2m2.dat', 'w')
    f111 = open(sphere + '/s0_c2m1.dat', 'w')
    f112 = open(sphere + '/s0_c20.dat', 'w')
    f113 = open(sphere + '/s0_c2p1.dat', 'w')
    f114 = open(sphere + '/s0_c2p2.dat', 'w')
 

    f120 = open(sphere + '/s0_c3m3.dat', 'w')
    f121 = open(sphere + '/s0_c3m2.dat', 'w')
    f122 = open(sphere + '/s0_c3m1.dat', 'w')
    f123 = open(sphere + '/s0_c30.dat', 'w')
    f124 = open(sphere + '/s0_c3p1.dat', 'w')
    f125 = open(sphere + '/s0_c3p2.dat', 'w')
    f126 = open(sphere + '/s0_c3p3.dat', 'w')

    f130 = open(sphere + '/s0_c4m4.dat', 'w')
    f131 = open(sphere + '/s0_c4m3.dat', 'w')
    f132 = open(sphere + '/s0_c4m2.dat', 'w')
    f133 = open(sphere + '/s0_c4m1.dat', 'w')
    f134 = open(sphere + '/s0_c40.dat', 'w')
    f135 = open(sphere + '/s0_c4p1.dat', 'w')
    f136 = open(sphere + '/s0_c4p2.dat', 'w')
    f137 = open(sphere + '/s0_c4p3.dat', 'w')
    f138 = open(sphere + '/s0_c4p4.dat', 'w')

    f400 = open(sphere + '/s0_dtc00.dat', 'w')
    f401 = open(sphere + '/s0_dtc20.dat', 'w')
    f402 = open(sphere + '/s0_dtc2p2.dat', 'w')

    h_phi      = np.pi/resolution_phi
    h_theta      = 2*np.pi/resolution_theta

    #------computing the spin-weighted spherical harmonics Y'_l,m------------
    #--  Y_l,m = Y'_l,m * exp ( +i*m*phi) ---------------------------------
    #--  Y_l,m = Y'_l,m * [cos( +i*m*phi) + i*sin( +i*m*phi)] -------------
    #-- *Y_l,m = Y'_l,m * [cos( +i*m*phi) - i*sin( +i*m*phi)] -------------
    Y00 = np.zeros((2, resolution_theta, resolution_phi))

    Y1m1 = np.zeros((2, resolution_theta, resolution_phi))
    Y10 = np.zeros((2, resolution_theta, resolution_phi))
    Y1p1 = np.zeros((2, resolution_theta, resolution_phi))

    Y2m2 = np.zeros((2, resolution_theta, resolution_phi))
    Y2m1 = np.zeros((2, resolution_theta, resolution_phi))
    Y20 = np.zeros((2, resolution_theta, resolution_phi))
    Y2p1 = np.zeros((2, resolution_theta, resolution_phi))
    Y2p2 = np.zeros((2, resolution_theta, resolution_phi))

    Y3m3 = np.zeros((2, resolution_theta, resolution_phi))
    Y3m2 = np.zeros((2, resolution_theta, resolution_phi))
    Y3m1 = np.zeros((2, resolution_theta, resolution_phi))
    Y30 = np.zeros((2, resolution_theta, resolution_phi))
    Y3p1 = np.zeros((2, resolution_theta, resolution_phi))
    Y3p2 = np.zeros((2, resolution_theta, resolution_phi))
    Y3p3 = np.zeros((2, resolution_theta, resolution_phi))

    Y4m4 = np.zeros((2, resolution_theta, resolution_phi))
    Y4m3 = np.zeros((2, resolution_theta, resolution_phi))
    Y4m2 = np.zeros((2, resolution_theta, resolution_phi))
    Y4m1 = np.zeros((2, resolution_theta, resolution_phi))
    Y40 = np.zeros((2, resolution_theta, resolution_phi))
    Y4p1 = np.zeros((2, resolution_theta, resolution_phi))
    Y4p2 = np.zeros((2, resolution_theta, resolution_phi))
    Y4p3 = np.zeros((2, resolution_theta, resolution_phi))
    Y4p4 = np.zeros((2, resolution_theta, resolution_phi))

    ty00 = 0.5 * np.sqrt(1/np.pi)


    for iphi in range(resolution_phi):
        phi = iphi*h_phi
        for itheta in range(resolution_theta):
            theta = itheta*h_theta 

            Y00[0, itheta, iphi] = ty00 * np.cos(0.0*theta)
            Y00[1, itheta, iphi] = -ty00 * np.sin(0.0*theta)

            ty1m1 = 0.5 * np.sqrt(3/(2*np.pi)) * np.sin(phi)
            ty10 = 0.5 * np.sqrt(3/np.pi) * np.cos(phi)
            ty1p1 = -0.5 * np.sqrt(3/(2*np.pi)) * np.sin(phi)
            Y1m1[0, itheta, iphi] = ty1m1*np.cos(-1.0*theta)
            Y10[0, itheta, iphi] = ty10*np.cos(0.0*theta)
            Y1p1[0, itheta, iphi] = ty1p1*np.cos(1.0*theta)
            Y1m1[1, itheta, iphi] = ty1m1*np.sin(-1.0*theta)
            Y10[1, itheta, iphi] = ty10*np.sin(0.0*theta)
            Y1p1[1, itheta, iphi] = ty1p1*np.sin(1.0*theta)
      
            ty2m2 = 0.25 * np.sqrt(15/(2*np.pi)) * np.sin(phi) * np.sin(phi)
            ty2m1 = 0.5 * np.sqrt(15/(2*np.pi)) * np.sin(phi) * np.cos(phi)
            ty20 = 0.25 * np.sqrt(5/np.pi) * (3 * np.cos(phi) * np.cos(phi) - 1)
            ty2p1 = -0.5 * np.sqrt(15/(2*np.pi)) * np.sin(phi) * np.cos(phi)
            ty2p2 = 0.25 * np.sqrt(15/(2*np.pi)) * np.sin(phi) * np.sin(phi)
            Y2m2[0, itheta, iphi] = ty2m2 * np.cos(-2*theta)
            Y2m1[0, itheta, iphi] = ty2m1 * np.cos(-1*theta)
            Y20[0, itheta, iphi] = ty20 * np.cos(0*theta)
            Y2p1[0, itheta, iphi] = ty2p1 * np.cos(1*theta)
            Y2p2[0, itheta, iphi] = ty2p2 * np.cos(2*theta)
            Y2m2[1, itheta, iphi] = ty2m2 * np.sin(-2*theta)
            Y2m1[1, itheta, iphi] = ty2m1 * np.sin(-1*theta)
            Y20[1, itheta, iphi] = ty20 * np.sin(0*theta)
            Y2p1[1, itheta, iphi] = ty2p1 * np.sin(1*theta)
            Y2p2[1, itheta, iphi] = ty2p2 * np.sin(2*theta)

            ty3m3 = 0.125 * np.sqrt(35/np.pi) * np.sin(phi) * np.sin(phi) * np.sin(phi)
            ty3m2 = 0.25 * np.sqrt(105/(2*np.pi)) * np.sin(phi) * np.sin(phi) * np.cos(phi)
            ty3m1 = 0.125 * np.sqrt(21/np.pi) * np.sin(phi) * (5*np.cos(phi) * np.cos(phi) - 1)
            ty30 = 0.25 * np.sqrt(7/np.pi) * (5*np.cos(phi) * np.cos(phi) * np.cos(phi) - 3 * np.cos(phi))
            ty3p1 = -0.125 * np.sqrt(21/np.pi) * np.sin(phi) * (5*np.cos(phi) * np.cos(phi) - 1)
            ty3p2 = 0.25 * np.sqrt(105/(2*np.pi)) * np.sin(phi) * np.sin(phi) * np.cos(phi)
            ty3p3 = -0.125 * np.sqrt(35/np.pi) * np.sin(phi) * np.sin(phi) * np.sin(phi)
            Y3m3[0, itheta, iphi] = ty3m3 * np.cos(-3*theta)
            Y3m2[0, itheta, iphi] = ty3m2 * np.cos(-2*theta)
            Y3m1[0, itheta, iphi] = ty3m1 * np.cos(-1*theta)
            Y30[0, itheta, iphi] = ty30 * np.cos(0*theta)
            Y3p1[0, itheta, iphi] = ty3p1 * np.cos(1*theta)
            Y3p2[0, itheta, iphi] = ty3p2 * np.cos(2*theta)
            Y3p3[0, itheta, iphi] = ty3p3 * np.cos(3*theta)
            Y3m3[1, itheta, iphi] = ty3m3 * np.sin(-3*theta)
            Y3m2[1, itheta, iphi] = ty3m2 * np.sin(-2*theta)
            Y3m1[1, itheta, iphi] = ty3m1 * np.sin(-1*theta)
            Y30[1, itheta, iphi] = ty30 * np.sin(0*theta)
            Y3p1[1, itheta, iphi] = ty3p1 * np.sin(1*theta)
            Y3p2[1, itheta, iphi] = ty3p2 * np.sin(2*theta)
            Y3p3[1, itheta, iphi] = ty3p3 * np.sin(3*theta)

            ty4m4 = 0.1875 * np.sqrt(35/(2*np.pi)) * np.sin(phi) * np.sin(phi) * np.sin(phi) * np.sin(phi)
            ty4m3 = 0.375 * np.sqrt(35/np.pi) * np.sin(phi) * np.sin(phi) * np.sin(phi) * np.cos(phi)
            ty4m2 = 0.375 * np.sqrt(5/(2*np.pi)) * np.sin(phi) * np.sin(phi) * (7*np.cos(phi) * np.cos(phi) - 1)
            ty4m1 = 0.375 * np.sqrt(5/np.pi) * np.sin(phi) * (7*np.cos(phi) * np.cos(phi) * np.cos(phi) - 3 * np.cos(phi))
            ty40 = 0.1875 * np.sqrt(1/np.pi) * (35 * np.cos(phi) * np.cos(phi) * np.cos(phi) * np.cos(phi) - 30 * np.cos(phi) * np.cos(phi) + 3)
            ty4p1 = -0.375 * np.sqrt(5/np.pi) * np.sin(phi) * (7*np.cos(phi) * np.cos(phi) * np.cos(phi) - 3 * np.cos(phi))
            ty4p2 = 0.375 * np.sqrt(5/(2*np.pi)) * np.sin(phi) * np.sin(phi) * (7*np.cos(phi) * np.cos(phi) - 1)
            ty4p3 = -0.375 * np.sqrt(35/np.pi) * np.sin(phi) * np.sin(phi) * np.sin(phi) * np.cos(phi)
            ty4p4 = 0.1875 * np.sqrt(35/(2*np.pi)) * np.sin(phi) * np.sin(phi) * np.sin(phi) * np.sin(phi)
            Y4m4[0, itheta, iphi] = ty4m4 * np.cos(-4*theta)
            Y4m3[0, itheta, iphi] = ty4m3 * np.cos(-3*theta)
            Y4m2[0, itheta, iphi] = ty4m2 * np.cos(-2*theta)
            Y4m1[0, itheta, iphi] = ty4m1 * np.cos(-1*theta)
            Y40[0, itheta, iphi] = ty40 * np.cos(0*theta)
            Y4p1[0, itheta, iphi] = ty4p1 * np.cos(1*theta)
            Y4p2[0, itheta, iphi] = ty4p2 * np.cos(2*theta)
            Y4p3[0, itheta, iphi] = ty4p3 * np.cos(3*theta)
            Y4p4[0, itheta, iphi] = ty4p4 * np.cos(4*theta)
            Y4m4[1, itheta, iphi] = ty4m4 * np.sin(-4*theta)
            Y4m3[1, itheta, iphi] = ty4m3 * np.sin(-3*theta)
            Y4m2[1, itheta, iphi] = ty4m2 * np.sin(-2*theta)
            Y4m1[1, itheta, iphi] = ty4m1 * np.sin(-1*theta)
            Y40[1, itheta, iphi] = ty40 * np.sin(0*theta)
            Y4p1[1, itheta, iphi] = ty4p1 * np.sin(1*theta)
            Y4p2[1, itheta, iphi] = ty4p2 * np.sin(2*theta)
            Y4p3[1, itheta, iphi] = ty4p3 * np.sin(3*theta)
            Y4p4[1, itheta, iphi] = ty4p4 * np.sin(4*theta)


    c00 = np.zeros((2, len(times)))
    c20 = np.zeros((2, len(times)))
    c2p2 = np.zeros((2, len(times)))
    for t in times:
        time = t*dt
        print("time=", time)
        field = readfield(sphere + "/t" + str(t), 'phi22')

        temp1 = field*Y00[0,:]
        temp2 = field*Y00[1,:]
        c00_r = surface_integral2(temp1, h_theta, h_phi)
        c00_i = surface_integral2(temp2, h_theta, h_phi)
        c00[0, t/delta_output] = c00_r
        c00[1, t/delta_output] = c00_i


        temp1 = field*Y1m1[0,:]
        temp2 = field*Y1m1[1,:]
        c1m1_r = surface_integral2(temp1, h_theta, h_phi)
        c1m1_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y10[0,:]
        temp2 = field*Y10[1,:]
        c10_r = surface_integral2(temp1, h_theta, h_phi)
        c10_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y1p1[0,:]
        temp2 = field*Y1p1[1,:]
        c1p1_r = surface_integral2(temp1, h_theta, h_phi)
        c1p1_i = surface_integral2(temp2, h_theta, h_phi)


        temp1 = field*Y2m2[0,:]
        temp2 = field*Y2m2[1,:]
        c2m2_r = surface_integral2(temp1, h_theta, h_phi)
        c2m2_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y2m1[0,:]
        temp2 = field*Y2m1[1,:]
        c2m1_r = surface_integral2(temp1, h_theta, h_phi)
        c2m1_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y20[0,:]
        temp2 = field*Y20[1,:]
        c20_r = surface_integral2(temp1, h_theta, h_phi)
        c20_i = surface_integral2(temp2, h_theta, h_phi)
        c20[0, t/delta_output] = c20_r
        c20[1, t/delta_output] = c20_i


        temp1 = field*Y2p1[0,:]
        temp2 = field*Y2p1[1,:]
        c2p1_r = surface_integral2(temp1, h_theta, h_phi)
        c2p1_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y2p2[0,:]
        temp2 = field*Y2p2[1,:]
        c2p2_r = surface_integral2(temp1, h_theta, h_phi)
        c2p2_i = surface_integral2(temp2, h_theta, h_phi)
        c2p2[0, t/delta_output] = c2p2_r
        c2p2[1, t/delta_output] = c2p2_i



        temp1 = field*Y3m3[0,:]
        temp2 = field*Y3m3[1,:]
        c3m3_r = surface_integral2(temp1, h_theta, h_phi)
        c3m3_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y3m2[0,:]
        temp2 = field*Y3m2[1,:]
        c3m2_r = surface_integral2(temp1, h_theta, h_phi)
        c3m2_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y3m1[0,:]
        temp2 = field*Y3m1[1,:]
        c3m1_r = surface_integral2(temp1, h_theta, h_phi)
        c3m1_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y30[0,:]
        temp2 = field*Y30[1,:]
        c30_r = surface_integral2(temp1, h_theta, h_phi)
        c30_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y3p1[0,:]
        temp2 = field*Y3p1[1,:]
        c3p1_r = surface_integral2(temp1, h_theta, h_phi)
        c3p1_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y3p2[0,:]
        temp2 = field*Y3p2[1,:]
        c3p2_r = surface_integral2(temp1, h_theta, h_phi)
        c3p2_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y3p3[0,:]
        temp2 = field*Y3p3[1,:]
        c3p3_r = surface_integral2(temp1, h_theta, h_phi)
        c3p3_i = surface_integral2(temp2, h_theta, h_phi)


        temp1 = field*Y4m4[0,:]
        temp2 = field*Y4m4[1,:]
        c4m4_r = surface_integral2(temp1, h_theta, h_phi)
        c4m4_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y4m3[0,:]
        temp2 = field*Y4m3[1,:]
        c4m3_r = surface_integral2(temp1, h_theta, h_phi)
        c4m3_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y4m2[0,:]
        temp2 = field*Y4m2[1,:]
        c4m2_r = surface_integral2(temp1, h_theta, h_phi)
        c4m2_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y4m1[0,:]
        temp2 = field*Y4m1[1,:]
        c4m1_r = surface_integral2(temp1, h_theta, h_phi)
        c4m1_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y40[0,:]
        temp2 = field*Y40[1,:]
        c40_r = surface_integral2(temp1, h_theta, h_phi)
        c40_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y4p1[0,:]
        temp2 = field*Y4p1[1,:]
        c4p1_r = surface_integral2(temp1, h_theta, h_phi)
        c4p1_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y4p2[0,:]
        temp2 = field*Y4p2[1,:]
        c4p2_r = surface_integral2(temp1, h_theta, h_phi)
        c4p2_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y4p3[0,:]
        temp2 = field*Y4p3[1,:]
        c4p3_r = surface_integral2(temp1, h_theta, h_phi)
        c4p3_i = surface_integral2(temp2, h_theta, h_phi)

        temp1 = field*Y4p4[0,:]
        temp2 = field*Y4p4[1,:]
        c4p4_r = surface_integral2(temp1, h_theta, h_phi)
        c4p4_i = surface_integral2(temp2, h_theta, h_phi)

        norm = (c00_r*c00_r + c00_i*c00_i
            + c1m1_r*c1m1_r + c1m1_i*c1m1_i
            + c10_r*c10_r + c10_i*c10_i
            + c1p1_r*c1p1_r + c1p1_i*c1p1_i
            + c2m2_r*c2m2_r + c2m2_i*c2m2_i
            + c2m1_r*c2m1_r + c2m1_i*c2m1_i
            + c20_r*c20_r + c20_i*c20_i
            + c2p1_r*c2p1_r + c2p1_i*c2p1_i
            + c2p2_r*c2p2_r + c2p2_i*c2p2_i
            + c3m3_r*c3m3_r + c3m3_i*c3m3_i
            + c3m2_r*c3m2_r + c3m2_i*c3m2_i
            + c3m1_r*c3m1_r + c3m1_i*c3m1_i
            + c30_r*c30_r + c30_i*c30_i
            + c3p1_r*c3p1_r + c3p1_i*c3p1_i
            + c3p2_r*c3p2_r + c3p2_i*c3p2_i
            + c3p3_r*c3p3_r + c3p3_i*c3p3_i
            + c4m4_r*c4m4_r + c4m4_i*c4m4_i
            + c4m3_r*c4m3_r + c4m3_i*c4m3_i
            + c4m2_r*c4m2_r + c4m2_i*c4m2_i
            + c4m1_r*c4m1_r + c4m1_i*c4m1_i
            + c40_r*c40_r + c40_i*c40_i
            + c4p1_r*c4p1_r + c4p1_i*c4p1_i
            + c4p2_r*c4p2_r + c4p2_i*c4p2_i
            + c4p3_r*c4p3_r + c4p3_i*c4p3_i
            + c4p4_r*c4p4_r + c4p4_i*c4p4_i)

        norm = np.sqrt(norm)
        norm = 1.0

        c00_r = c00_r / norm
        c00_i = c00_i / norm
        c1m1_r = c1m1_r / norm
        c1m1_i = c1m1_i / norm
        c10_r = c10_r / norm
        c10_i = c10_i / norm
        c1p1_r = c1p1_r / norm
        c1p1_i = c1p1_i / norm
        c2m2_r = c2m2_r / norm
        c2m2_i = c2m2_i / norm
        c2m1_r = c2m1_r / norm
        c2m1_i = c2m1_i / norm
        c20_r = c20_r / norm
        c20_i = c20_i / norm
        c2p1_r = c2p1_r / norm
        c2p1_i = c2p1_i / norm
        c2p2_r = c2p2_r / norm
        c2p2_i = c2p2_i / norm
        c3m3_r = c3m3_r / norm
        c3m3_i = c3m3_i / norm
        c3m2_r = c3m2_r / norm
        c3m2_i = c3m2_i / norm
        c3m1_r = c3m1_r / norm
        c3m1_i = c3m1_i / norm
        c30_r = c30_r / norm
        c30_i = c30_i / norm
        c3p1_r = c3p1_r /norm
        c3p1_i = c3p1_i / norm
        c3p2_r = c3p2_r / norm
        c3p2_i = c3p2_i / norm
        c3p3_r = c3p3_r / norm
        c3p3_i = c3p3_i / norm
        c4m4_r = c4m4_r / norm
        c4m4_i = c4m4_i / norm
        c4m3_r = c4m3_r / norm
        c4m3_i = c4m3_i / norm
        c4m2_r = c4m2_r /norm
        c4m2_i = c4m2_i / norm
        c4m1_r = c4m1_r / norm
        c4m1_i = c4m1_i / norm
        c40_r = c40_r / norm
        c40_i = c40_i / norm
        c4p1_r = c4p1_r / norm
        c4p1_i = c4p1_i / norm
        c4p2_r = c4p2_r / norm
        c4p2_i = c4p2_i / norm
        c4p3_r = c4p3_r / norm 
        c4p3_i = c4p3_i / norm
        c4p4_r = c4p4_r / norm
        c4p4_i = c4p4_i / norm

        print("-------------------------")
        print("-----L=0 MODES ----------")
        print("-------------------------")
        print("c00=",c00_r,  "c00=",c00_i)

        f90.write(str(time) + " " + str(c00_r) + " " + str(c00_i) + "\n")

        print("-------------------------")
        print("-----L=1 MODES ----------")
        print("-------------------------")
        print("c1m1=",c1m1_r,"c1m1=",c1m1_i)
        print("c10=",c10_r,  "c10=",c10_i)
        print("c1p1=",c1p1_r,"c1p1=",c1p1_i)

        f100.write(str(time) + " " + str(c1m1_r) + " " + str(c1m1_i) + "\n")
        f101.write(str(time) + " " + str(c10_r) + " " + str(c10_i) + "\n")
        f102.write(str(time) + " " + str(c1p1_r) + " " + str(c1p1_i) + "\n")

        print("-------------------------")   
        print("-----L=2 MODES ----------")
        print("-------------------------")
        print("c2m2=",c2m2_r,"c2m2=",c2m2_i)
        print("c2m1=",c2m1_r,"c2m1=",c2m1_i)
        print("c20=",c20_r,  "c20=",c20_i)
        print("c2p1=",c2p1_r,"c2p1=",c2p1_i)
        print("c2p2=",c2p2_r,"c2p2=",c2p2_i)

        f110.write(str(time) + " " + str(c2m2_r) + " " + str(c2m2_i) + "\n")
        f111.write(str(time) + " " + str(c2m1_r) + " " + str(c2m1_i) + "\n")
        f112.write(str(time) + " " + str(c20_r) + " " + str(c20_i) + "\n")
        f113.write(str(time) + " " + str(c2p1_r) + " " + str(c2p1_i) + "\n")
        f114.write(str(time) + " " + str(c2p2_r) + " " + str(c2p2_i) + "\n")

        print("-------------------------")
        print("-----L=3 MODES ----------")
        print("-------------------------")
        print("c3m3=",c3m3_r,"c3m3=",c3m3_i)
        print("c3m2=",c3m2_r,"c3m2=",c3m2_i)
        print("c3m1=",c3m1_r,"c3m1=",c3m1_i)
        print("c30=",c30_r,  "c30=",c30_i)
        print("c3p1=",c3p1_r,"c3p1=",c3p1_i)
        print("c3p2=",c3p2_r,"c3p2=",c3p2_i)
        print("c3p3=",c3p3_r,"c3p3=",c3p3_i)

        f120.write(str(time) + " " + str(c3m3_r) + " " + str(c3m3_i) + "\n")
        f121.write(str(time) + " " + str(c3m2_r) + " " + str(c3m2_i) + "\n")
        f122.write(str(time) + " " + str(c3m1_r) + " " + str(c3m1_i) + "\n")
        f123.write(str(time) + " " + str(c30_r) + " " + str(c30_i) + "\n")
        f124.write(str(time) + " " + str(c3p1_r) + " " + str(c3p1_i) + "\n")
        f125.write(str(time) + " " + str(c3p2_r) + " " + str(c3p2_i) + "\n")
        f126.write(str(time) + " " + str(c3p3_r) + " " + str(c3p3_i) + "\n")

        print("-------------------------")
        print("-----L=4 MODES ----------")
        print("-------------------------")
        print("c4m4=",c4m4_r,"c4m4=",c4m4_i)
        print("c4m3=",c4m3_r,"c4m3=",c4m3_i)
        print("c4m2=",c4m2_r,"c4m2=",c4m2_i)
        print("c4m1=",c4m1_r,"c4m1=",c4m1_i)
        print("c40=",c40_r,  "c40=",c40_i)
        print("c4p1=",c4p1_r,"c4p1=",c4p1_i)
        print("c4p2=",c4p2_r,"c4p2=",c4p2_i)
        print("c4p3=",c4p3_r,"c4p3=",c4p3_i)
        print("c4p4=",c4p4_r,"c4p4=",c4p4_i)

        f130.write(str(time) + " " + str(c4m4_r) + " " + str(c4m4_i) + "\n")
        f131.write(str(time) + " " + str(c4m3_r) + " " + str(c4m3_i) + "\n")
        f132.write(str(time) + " " + str(c4m2_r) + " " + str(c4m2_i) + "\n")
        f133.write(str(time) + " " + str(c4m1_r) + " " + str(c4m1_i) + "\n")
        f134.write(str(time) + " " + str(c40_r) + " " + str(c40_i) + "\n")
        f135.write(str(time) + " " + str(c4p1_r) + " " + str(c4p1_i) + "\n")
        f136.write(str(time) + " " + str(c4p2_r) + " " + str(c4p2_i) + "\n")
        f137.write(str(time) + " " + str(c4p3_r) + " " + str(c4p3_i) + "\n")
        f138.write(str(time) + " " + str(c4p4_r) + " " + str(c4p4_i) + "\n")


    f90.close()
    f100.close()
    f101.close()
    f102.close()

    f110.close()
    f111.close()
    f112.close()
    f113.close()
    f114.close()

    f120.close()
    f121.close()
    f122.close()
    f123.close()
    f124.close()
    f125.close()
    f126.close()

    f130.close()
    f131.close()
    f132.close()
    f133.close()
    f134.close()
    f135.close()
    f136.close()
    f137.close()
    f138.close()



    dtc00 = np.zeros((2, len(times) - 2))
    dtc20 = np.zeros((2, len(times) - 2))
    dtc2p2 = np.zeros((2, len(times) - 2))
    dt2 = 3.51564

    for i in xrange(1, len(times) - 1):
        time = i*dt2
        print "i=", i,"time=", time

        dtc00 = (c00[:, i-1] + c00[:, i+1] - 2 * c00[:, i]) / dt2
        f400.write(str(time) + " " + str(dtc00[0]) + " " + str(dtc00[1]) + "\n")
        dtc20 = (c20[:, i-1] + c20[:, i+1] - 2 * c20[:, i]) / dt2
        f401.write(str(time) + " " + str(dtc20[0]) + " " + str(dtc20[1]) + "\n")
        dtc2p2 = (c2p2[:, i-1] + c2p2[:, i+1] - 2 * c2p2[:, i]) / dt2
        f402.write(str(time) + " " + str(dtc2p2[0]) + " " + str(dtc2p2[1]) + "\n")

    f400.close()
    f401.close()
    f402.close()

if __name__ == "__main__":
    for (sphere, rext) in [('data_outputDir_sphere0', 0),('data_outputDir_sphere1', 1),('data_outputDir_sphere2', 2),('data_outputDir_sphere3', 3),('data_outputDir_sphere4', 4),('data_outputDir_sphere5', 5) ]:
        calculate(sphere, rext)
