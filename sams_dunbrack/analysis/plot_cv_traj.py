from features import featurize
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

(key_res, dihedrals, distances) = featurize.featurize(chain='A', coord='dcd', feature='conf', pdb='5UG9')
#print(key_res)
dih = {}; dis = {}
for i in range(10):
    dih[i] = dihedrals[:,i]

for i in range(5):
    dis[i] = distances[:,i]
#baseline values
# state 0: BLAminus
dunbrack_phi0 = dict(); dunbrack_dphi = dict()
dunbrack_r0 = dict(); dunbrack_dr = dict()

dunbrack_phi0[0,0] = -120.44; dunbrack_dphi[0,0] = 32.55
dunbrack_phi0[0,1] = -129.94; dunbrack_dphi[0,1] = 9.88
dunbrack_phi0[0,2] = 5.02; dunbrack_dphi[0,2] = 172.32
dunbrack_phi0[0,3] = 59.72; dunbrack_dphi[0,3] = 13.47
dunbrack_phi0[0,4] = 82.29; dunbrack_dphi[0,4] = 11.15
dunbrack_phi0[0,5] = -141.98; dunbrack_dphi[0,5] = 54.98
dunbrack_phi0[0,6] = 1.98; dunbrack_dphi[0,6] = 69.82
dunbrack_phi0[0,7] = -97.99; dunbrack_dphi[0,7] = 9.86
dunbrack_phi0[0,8] = 21.59; dunbrack_dphi[0,8] = 9.46
dunbrack_phi0[0,9] = -69.92; dunbrack_dphi[0,9] = 9.62
dunbrack_r0[0,0] = 0.36; dunbrack_dr[0,0] = 0.12
dunbrack_r0[0,1] = 0.35; dunbrack_dr[0,1] = 0.13
dunbrack_r0[0,2] = 0.65; dunbrack_dr[0,2] = 0.10
dunbrack_r0[0,3] = 1.45; dunbrack_dr[0,3] = 0.08
dunbrack_r0[0,4] = 2.82; dunbrack_dr[0,4] = 0.98
# state 1: BLAplus
dunbrack_phi0[1,0] = -130.24; dunbrack_dphi[1,0] = 12.29
dunbrack_phi0[1,1] = -124.32; dunbrack_dphi[1,1] = 11.43
dunbrack_phi0[1,2] = 72.42; dunbrack_dphi[1,2] = 152.36
dunbrack_phi0[1,3] = 57.33; dunbrack_dphi[1,3] = 17.31
dunbrack_phi0[1,4] = 34.54; dunbrack_dphi[1,4] = 14.41
dunbrack_phi0[1,5] = -90.96; dunbrack_dphi[1,5] = 46.94
dunbrack_phi0[1,6] = -11.83; dunbrack_dphi[1,6] = 75.75
dunbrack_phi0[1,7] = -90.69; dunbrack_dphi[1,7] = 18.78
dunbrack_phi0[1,8] = -10.37; dunbrack_dphi[1,8] = 17.96
dunbrack_phi0[1,9] = 54.04; dunbrack_dphi[1,9] = 11.30
dunbrack_r0[1,0] = 0.62; dunbrack_dr[1,0] = 0.38
dunbrack_r0[1,1] = 0.63; dunbrack_dr[1,1] = 0.38
dunbrack_r0[1,2] = 0.54; dunbrack_dr[1,2] = 0.11
dunbrack_r0[1,3] = 1.35; dunbrack_dr[1,3] = 0.06
dunbrack_r0[1,4] = 2.01; dunbrack_dr[1,4] = 0.80
# state 2: ABAminus
dunbrack_phi0[2,0] = -127.64; dunbrack_dphi[2,0] = 11.80
dunbrack_phi0[2,1] = -109.30; dunbrack_dphi[2,1] = 18.41
dunbrack_phi0[2,2] = -20.49; dunbrack_dphi[2,2] = 28.27
dunbrack_phi0[2,3] = -129.76; dunbrack_dphi[2,3] = 22.41
dunbrack_phi0[2,4] = 114.93; dunbrack_dphi[2,4] = 87.01
dunbrack_phi0[2,5] = -32.89; dunbrack_dphi[2,5] = 140.66
dunbrack_phi0[2,6] = 5.92; dunbrack_dphi[2,6] = 71.05
dunbrack_phi0[2,7] = -119.31; dunbrack_dphi[2,7] = 14.66
dunbrack_phi0[2,8] = 19.19; dunbrack_dphi[2,8] = 11.99
dunbrack_phi0[2,9] = -61.56; dunbrack_dphi[2,9] = 9.72
dunbrack_r0[2,0] = 0.45; dunbrack_dr[2,0] = 0.16
dunbrack_r0[2,1] = 0.42; dunbrack_dr[2,1] = 0.17
dunbrack_r0[2,2] = 0.56; dunbrack_dr[2,2] = 0.12
dunbrack_r0[2,3] = 1.43; dunbrack_dr[2,3] = 0.08
dunbrack_r0[2,4] = 2.76; dunbrack_dr[2,4] = 0.93
# state 3: BLBminus
dunbrack_phi0[3,0] = -112.29; dunbrack_dphi[3,0] = 21.14
dunbrack_phi0[3,1] = -133.59; dunbrack_dphi[3,1] = 9.58
dunbrack_phi0[3,2] = 42.96; dunbrack_dphi[3,2] = 163.61
dunbrack_phi0[3,3] = 57.39; dunbrack_dphi[3,3] = 57.39
dunbrack_phi0[3,4] = 63.69; dunbrack_dphi[3,4] = 15.45
dunbrack_phi0[3,5] = -129.99; dunbrack_dphi[3,5] = 54.86
dunbrack_phi0[3,6] = -6.17; dunbrack_dphi[3,6] = 81.33
dunbrack_phi0[3,7] = -74.52; dunbrack_dphi[3,7] = 23.84
dunbrack_phi0[3,8] = 129.52; dunbrack_dphi[3,8] = 65.69
dunbrack_phi0[3,9] = -71.17; dunbrack_dphi[3,9] = 11.69
dunbrack_r0[3,0] = 0.78; dunbrack_dr[3,0] = 0.51
dunbrack_r0[3,1] = 0.80; dunbrack_dr[3,1] = 0.53
dunbrack_r0[3,2] = 0.73; dunbrack_dr[3,2] = 0.12
dunbrack_r0[3,3] = 1.55; dunbrack_dr[3,3] = 0.08
dunbrack_r0[3,4] = 2.36; dunbrack_dr[3,4] = 0.49
# state 4: BLBplus
dunbrack_phi0[4,0] = -113.89; dunbrack_dphi[4,0] = 14.01
dunbrack_phi0[4,1] = -128.11; dunbrack_dphi[4,1] = 12.51
dunbrack_phi0[4,2] = 95.94; dunbrack_dphi[4,2] = 135.52
dunbrack_phi0[4,3] = 61.06; dunbrack_dphi[4,3] = 8.41
dunbrack_phi0[4,4] = 34.09; dunbrack_dphi[4,4] = 14.94
dunbrack_phi0[4,5] = -85.46; dunbrack_dphi[4,5] = 59.67
dunbrack_phi0[4,6] = -1.76; dunbrack_dphi[4,6] = 65.61
dunbrack_phi0[4,7] = -89.76; dunbrack_dphi[4,7] = 28.00
dunbrack_phi0[4,8] = 139.64; dunbrack_dphi[4,8] = 47.62
dunbrack_phi0[4,9] = 49.29; dunbrack_dphi[4,9] = 17.81
dunbrack_r0[4,0] = 1.10; dunbrack_dr[4,0] = 0.51
dunbrack_r0[4,1] = 1.13; dunbrack_dr[4,1] = 0.57
dunbrack_r0[4,2] = 0.58; dunbrack_dr[4,2] = 0.09
dunbrack_r0[4,3] = 1.38; dunbrack_dr[4,3] = 0.08
dunbrack_r0[4,4] = 2.45; dunbrack_dr[4,4] = 0.88
# state 5: BLBtrans
dunbrack_phi0[5,0] = -98.33; dunbrack_dphi[5,0] = 6.17
dunbrack_phi0[5,1] = -111.12; dunbrack_dphi[5,1] = 10.67
dunbrack_phi0[5,2] = 139.20; dunbrack_dphi[5,2] = 69.16
dunbrack_phi0[5,3] = 69.31; dunbrack_dphi[5,3] = 7.11
dunbrack_phi0[5,4] = 23.98; dunbrack_dphi[5,4] = 8.03
dunbrack_phi0[5,5] = -71.93; dunbrack_dphi[5,5] = 25.78
dunbrack_phi0[5,6] = 7.41; dunbrack_dphi[5,6] = 63.29
dunbrack_phi0[5,7] = -65.50; dunbrack_dphi[5,7] = 10.18
dunbrack_phi0[5,8] = 132.51; dunbrack_dphi[5,8] = 23.15
dunbrack_phi0[5,9] = -79.15; dunbrack_dphi[5,9] = 112.45
dunbrack_r0[5,0] = 1.59; dunbrack_dr[5,0] = 0.17
dunbrack_r0[5,1] = 1.61; dunbrack_dr[5,1] = 0.19
dunbrack_r0[5,2] = 0.98; dunbrack_dr[5,2] = 0.21
dunbrack_r0[5,3] = 1.79; dunbrack_dr[5,3] = 0.08
dunbrack_r0[5,4] = 2.37; dunbrack_dr[5,4] = 0.33

# state 6: BBAminus
dunbrack_phi0[6,0] = -126.27; dunbrack_dphi[6,0] = 13.24
dunbrack_phi0[6,1] = -139.03; dunbrack_dphi[6,1] = 24.21
dunbrack_phi0[6,2] = -68.84; dunbrack_dphi[6,2] = 156.12
dunbrack_phi0[6,3] = -140.53; dunbrack_dphi[6,3] = 16.48
dunbrack_phi0[6,4] = 102.15; dunbrack_dphi[6,4] = 16.29
dunbrack_phi0[6,5] = -123.39; dunbrack_dphi[6,5] = 115.30
dunbrack_phi0[6,6] = 16.10; dunbrack_dphi[6,6] = 73.16
dunbrack_phi0[6,7] = -83.02; dunbrack_dphi[6,7] = 17.06
dunbrack_phi0[6,8] = -9.52; dunbrack_dphi[6,8] = 24.54
dunbrack_phi0[6,9] = -68.50; dunbrack_dphi[6,9] = 17.04
dunbrack_r0[6,0] = 0.44; dunbrack_dr[6,0] = 0.28
dunbrack_r0[6,1] = 0.45; dunbrack_dr[6,1] = 0.31
dunbrack_r0[6,2] = 1.40; dunbrack_dr[6,2] = 0.13
dunbrack_r0[6,3] = 1.05; dunbrack_dr[6,3] = 0.12
dunbrack_r0[6,4] = 2.35; dunbrack_dr[6,4] = 0.76
# state 7: BABtrans
dunbrack_phi0[7,0] = -117.01; dunbrack_dphi[7,0] = 10.17
dunbrack_phi0[7,1] = -77.79; dunbrack_dphi[7,1] = 14.68
dunbrack_phi0[7,2] = 133.19; dunbrack_dphi[7,2] = 8.66
dunbrack_phi0[7,3] = -107.14; dunbrack_dphi[7,3] = 18.61
dunbrack_phi0[7,4] = 5.84; dunbrack_dphi[7,4] = 29.51
dunbrack_phi0[7,5] = 44.57; dunbrack_dphi[7,5] = 50.33
dunbrack_phi0[7,6] = 22.60; dunbrack_dphi[7,6] = 110.27
dunbrack_phi0[7,7] = -72.94; dunbrack_dphi[7,7] = 17.33
dunbrack_phi0[7,8] = 130.51; dunbrack_dphi[7,8] = 13.76
dunbrack_phi0[7,9] = -111.22; dunbrack_dphi[7,9] = 97.45
dunbrack_r0[7,0] = 0.72; dunbrack_dr[7,0] = 0.35
dunbrack_r0[7,1] = 0.76; dunbrack_dr[7,1] = 0.40
dunbrack_r0[7,2] = 0.89; dunbrack_dr[7,2] = 0.12
dunbrack_r0[7,3] = 0.65; dunbrack_dr[7,3] = 0.16
dunbrack_r0[7,4] = 2.94; dunbrack_dr[7,4] = 0.79

#change degrees to radians
dunbrack_phi0.update((x, y/180*np.pi) for x, y in dunbrack_phi0.items())
dunbrack_dphi.update((x, y/180*np.pi) for x, y in dunbrack_dphi.items())

#define colors
c = {}
c[0] = '#FF0000' #red
c[1] = '#FF8C00' #orange
c[2] = '#FFD700' #yellow
c[3] = '#32CD32' #green
c[4] = '#48D1CC' #teal
c[5] = '#0000FF' #blue
c[6] = '#8A2BE2' #magenta
c[7] = '#FF1493' #pink

fig, ax = plt.subplots(nrows=3, ncols=5)
r_count = 0
for row in ax[0:2]:
    # dihedrals
    c_count = 0
    for col in row:
        index = c_count+r_count*5
        col.scatter(list(range(len(dih[index]))), dih[index], s = 10, c='#8785a2')
        col.set_ylim(-3.5, 3.5)        
        for i in range(8): #for each dihedral
            col.axhline(dunbrack_phi0[i,index], color=c[i], ls='-')
            #col.fill_between(list(range(len(dih[index]))), dunbrack_phi0[i,index] - dunbrack_dphi[i,index], dunbrack_phi0[i,index] + dunbrack_dphi[i,index], color=c[i], alpha=0.3)
        c_count += 1
    r_count += 1
# distances
c_count = 0
for col in ax[2]:
    col.scatter(list(range(len(dis[c_count]))), dis[c_count], s = 10, c='#8785a2')
    col.set_ylim(0.2, 5)
    for i in range(8):  # for each dihedral
        col.axhline(dunbrack_r0[i, c_count], color=c[i], ls='-')
        #col.fill_between(list(range(len(dis[c_count]))), dunbrack_r0[i,c_count] - dunbrack_dr[i,c_count], dunbrack_r0[i,c_count] + dunbrack_dr[i,c_count], color=c[i], alpha=0.3)
    c_count += 1

#plt.scatter(list(range(len(data))), data, s=10, c='b')
plt.show()
