from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lapack




# ****************************************

def grid_for_1D_Bsplines(n_inter,LegPoints,x_begin,x_end) :
  #x_grid,x_weigths=np.zeros((LegPoints,n_inter))
  #x_sectors=np.zeros((n_inter+1))
  dx=(x_end-x_begin)/n_inter
  for i in range(n_inter):
    x_sectors[i]=x_begin+dx*i
    coef=dx/2.
    x=np.polynomial.legendre.leggauss(LegPoints)[0]
    x_grid[i,:]=coef*(x[:]+1.)+x_sectors[i]
    x_weigths[i,:]=coef*np.polynomial.legendre.leggauss(LegPoints)[1]
  x_sectors[n_inter]=x_end
  return x_grid,x_weigths,x_sectors


# ****************************************

class BasisFunc:
  'basis function class'
  basisCount = 0

  def __init__(self, b_coef):
    BasisFunc.basisCount += 1
    self.b_coef = b_coef
    
#    self.first_sector=-1
    for k_sec in range(n_inter+order):
      if(b_coef[k_sec]>0.):
        self.first_sector = max(0,k_sec-order)
        break
    for k_sec in range(n_inter+order-1,0,-1):
      if(b_coef[k_sec]>0.):
        self.last_sector = min(n_inter-1,k_sec)
        break
        
#    print ('checking atributes',BasisFunc.basisCount,n_inter,self.first_sector,self.last_sector,b_coef)
    f_on_grid =np.zeros([order+1,LegPoints],dtype =float)
    f_on_grid_der =np.zeros([order+1,LegPoints],dtype =float)
    for k_sec in range(order+1) : #  !!! change 'order+1' to 'order' ?
      k_inter=self.first_sector+k_sec
      if(k_inter<n_inter) :
        spl = BSpline(t_knots, b_coef, order)
        #deriv
        t1, c1, k1=BSpline(t_knots, b_coef, order).derivative().tck
        spld=BSpline(t1, c1, k1)
#       print ('checking atributes2',b_coef,c1)

        for l in range(LegPoints) :
          f_on_grid[k_sec,l]    = spl (x_grid[k_inter,l])
          f_on_grid_der[k_sec,l]= spld(x_grid[k_inter,l])
    self.f_on_grid = f_on_grid
    self.f_on_grid_der = f_on_grid_der
   
# ****************************************

def constructing_basis_old(boundary_left,boundary_right) :
  basis_dim=n_inter+order
  if(boundary_left>0  ) : basis_dim=basis_dim-1
  if(boundary_left==3 ) : basis_dim=basis_dim-1
  if(boundary_right>0 ) : basis_dim=basis_dim-1
  if(boundary_right==3) : basis_dim=basis_dim-1
  bf = [] 
  n_b=0
  c0=np.zeros([n_inter+2*order],dtype =float)
  for i in range(n_inter+order) :
    if(i==0               and boundary_left==1 ) : continue
    if(i==1               and boundary_left==2 ) : continue
    if(i==n_inter+order-1 and boundary_left==3 ) : continue
    if(i==n_inter+order-2 and boundary_left==4 ) : continue
    n_b=n_b+1
    c0[0:n_inter+2*order] =0.; c0[i]=1.
    bf1=BasisFunc(c0)    
    bf.append(bf1)
  return basis_dim,bf
# ****************************************

def constructing_basis(boundary_left,boundary_right) :
  basis_dim=n_inter+order
  if(boundary_left>0  ) : basis_dim=basis_dim-1
  if(boundary_left==3 ) : basis_dim=basis_dim-1
  if(boundary_right>0 ) : basis_dim=basis_dim-1
  if(boundary_right==3) : basis_dim=basis_dim-1
  bf = [] 
  n_b=0
  c0=np.zeros([n_inter+2*order],dtype =float)
#  print('basis_dim=',basis_dim)
  
  for j in range(basis_dim) :
    c0[0:n_inter+2*order] =0.
    if(boundary_left==0 ) : #no conditions on the left, i=j and c0[1]=1
      c0[j]=1. 
      if(j==0) : c0[1]=1.
    if(boundary_left==1 ) : #psi(left)=0, i=j+1 
      c0[j+1]=1.
    if(boundary_left==2 ) : #psi'(left)=0, i=j+1 and c0[1]=1
      c0[j+1]=1.
      if(j==0) : c0[0]=1.
    if(boundary_left==3 ) : #psi(left)=0, psi'(left)=0, i=j+2
      c0[j+2]=1. 
      #if no conditions on the right or psi'(left)=0,  only the last function is modified
    if((boundary_right==0 or boundary_right==2) and j==basis_dim-1) : 
      c0[n_inter+order-2:n_inter+order]=1.

    n_b=n_b+1
    bf1=BasisFunc(c0)    
    bf.append(bf1)
  return basis_dim,bf
# ****************************************
def integral_u_u(bf1,bf2,f1) :
  low_limit=max(bf1.first_sector,bf2.first_sector)
  upper_limit=min(bf1.last_sector,bf2.last_sector)
  overlap=0.
  for i in range(low_limit,upper_limit+1):
    for l in range(LegPoints):
      overlap=overlap+ bf1.f_on_grid[i-bf1.first_sector,l]*bf2.f_on_grid[i-bf2.first_sector,l]*x_weigths[i,l]*f1[i,l]

  return overlap,low_limit,upper_limit
# ****************************************
def integral_u_upp(bf1,bf2) :
  low_limit=max(bf1.first_sector,bf2.first_sector)
  upper_limit=min(bf1.last_sector,bf2.last_sector)
  overlap=0.
  for i in range(low_limit,upper_limit+1):
    for l in range(LegPoints):
      overlap=overlap+ bf1.f_on_grid_der[i-bf1.first_sector,l]*x_weigths[i,l]*bf2.f_on_grid_der[i-bf2.first_sector,l]
# add boundary terms here!!
  overlap=-overlap
  return overlap,low_limit,upper_limit
# ****************************************
def potential(r):
#  l=1;pot=-1/r+l*(l+1)/(2.*mass*r*r) #_modified Coulomb potential
#  l=5;pot=40000/r**12-2000/r**6+l*(l+1)/(2.*mass*r*r) #_modified Coulomb potential
#  pot=(r-5.)**2/2                     # harmonic oscillator potential
#  pot=0.                             # square-well potential
  
  # pot=9*((1-np.exp(-1*(r-5.)))**2-1.) + 300./(2.*r*r) # Morse
  # l_absorb = 60
  # A5 = 0.7




  R = 7.42 * 1.8897e-5
  Z = 84
  V0 = 62 * 36749.7
  sigma = 0.68 * 1.8897e-5

  l_absorb = 100 * 1.8897e-5
  # strength_absorb = 1
  A5 = 0.7 * 36749.7

  V_WS   = - V0/(1+ np.exp((r-R)/sigma))
  # print("r=",r, "R=", R, "(r-R)/sigma", (r-R)/sigma)
  # V_Wall = V1*exp(-r_fm/delta)

  V_C = np.zeros_like(r)
  cond = r <= R
  V_C[cond] = 2*(Z-2)*(3*R**2-r[cond]**2)/(2*R**3)
  V_C[~cond] = 2*(Z-2)/r[~cond]

  pot  = V_WS + V_C
  # pot[-1] = 1e10
  # pot[0] = 1e10



  if( resonances == 1 ):
    pot = pot.astype(np.complex128)
    r0 = x_sectors[n_inter] - l_absorb
    # print(f"r0 = {r0}")

    cond = r > r0
#      pot=pot-1j*strength_absorb*(r-(x_end-l_absorb))**2
    # pot[cond] -= 1j * A5 * 13. * np.exp( -2 * l_absorb/( r[cond] - r0 ) )
    pot[cond] -= 1j * A5 * 13. * np.exp( -2 * l_absorb/( r[cond] - r0 ) )

  
  return pot

# ****************************************
def plotting_pot(E=None):
  plt.title("potential")

#  c0[0:n_inter+2*order] =0.; c[i]=1.1
  npoints_for_plot=1000
  # x_new = np.linspace(min(t_knots), max(t_knots), npoints_for_plot)
  x_new = t_knots
  
  V = potential(x_new)

  plt.plot(x_new, V.real/36749.7, '-r', label="potential")
  if resonances:
    plt.plot(x_new, V.imag/36749.7, '-b', label="imaginary part of potential")
  plt.grid()
  plt.legend(loc='best', fancybox=True, shadow=True)
#  plt.ylim(-0.4, 0.1)# modified Coulomb potential

  # plt.ylim(-6, 6)

  # plt.xscale('log', base=10)

  if E is not None:
    plt.hlines(E, xmin=min(t_knots), xmax=max(t_knots), colors='g', linestyles='solid')

  plt.ylabel("energies")
  plt.xlabel("coordinate")

  plt.show()

# ****************************************
def construct_hamiltonian(hamiltonian,overlap_mat):
  f1 = np.ones([n_inter,LegPoints],dtype =float)
  #potential function
  v_pot = potential(x_grid)
#   if(resonances == 0):
#     v_pot = np.zeros([n_inter,LegPoints],dtype =float)
#   else:
#     v_pot=np.zeros([n_inter,LegPoints],dtype =complex)
#   for i in range(n_inter):
#     for l in range(LegPoints):
# #      v_pot[i,l]=(x_grid[i,l]-5)*(x_grid[i,l]-5)/2.
#       v_pot[i,l]=potential(x_grid[i,l])
     
  for i in range(basis_dim):
    for j in range(basis_dim):
       hamiltonian[i,j]=integral_u_upp(bf[i],bf[j])[0]/(-2.*mass)+integral_u_u(bf[i],bf[j],v_pot)[0]
       overlap_mat[i,j]=integral_u_u(bf[i],bf[j],f1)[0]
  return 
# ****************************************

def plotting_wf(eigen_vec,iv) :
  c0=np.zeros([n_inter+2*order],dtype =float)

  if(boundary_left==0 ) : #no conditions on the left, i=j and c0[1]=1
    c0[0:basis_dim]=eigen_vec[0:basis_dim,iv]
  if(boundary_left==1 ) : #psi(left)=0, i=j+1 
    c0[1:basis_dim+1]=eigen_vec[0:basis_dim,iv]
  if(boundary_left==2 ) : #psi'(left)=0, i=j+1 and c0[1]=1
    c0[1:basis_dim+1]=eigen_vec[0:basis_dim,iv]
    c0[0]= c0[1]
  if(boundary_left==3 ) : #psi(left)=0, psi'(left)=0, i=j+2
    c0[2:basis_dim+2]=eigen_vec[0:basis_dim,iv] # needs to be checked
      #if no conditions on the right or psi'(left)=0,  only the last function is modified
  if(boundary_right==2) : # needs to be checked
    c0[nbasis_dim-1]=c0[basis_dim-2] # needs to be checked


  plt.title("wave function, v="+ " {}".format(iv))

#  c0[0:n_inter+2*order] =0.; c[i]=1.
  npoints_for_plot=1000
  spl = BSpline(t_knots, c0, order)
  x_new = np.linspace(min(t_knots), max(t_knots), npoints_for_plot)
  y_fit = BSpline(t_knots, c0, order)(x_new)
  plt.plot(x_new, y_fit, '-b', label="v="+ " {}".format(iv))
  plt.legend(loc='best', fancybox=True, shadow=True)
  plt.grid()
  plt.show() 
# ****************************************

def plotting_wf_c(eigen_vec,iv) :
  c0=np.zeros([n_inter+2*order],dtype =float)
  c1=np.zeros([n_inter+2*order],dtype =float)

  if(boundary_left==0 ) : #no conditions on the left, i=j and c0[1]=1
    c0[0:basis_dim]=eigen_vec[0:basis_dim,iv].real
    c1[0:basis_dim]=eigen_vec[0:basis_dim,iv].imag
  if(boundary_left==1 ) : #psi(left)=0, i=j+1 
    c0[1:basis_dim+1]=eigen_vec[0:basis_dim,iv].real
    c1[1:basis_dim+1]=eigen_vec[0:basis_dim,iv].imag
  if(boundary_left==2 ) : #psi'(left)=0, i=j+1 and c0[1]=1
    c0[1:basis_dim+1]=eigen_vec[0:basis_dim,iv].real
    c1[1:basis_dim+1]=eigen_vec[0:basis_dim,iv].imag
    c0[0]= c0[1]
    c1[0]= c1[1]
  if(boundary_left==3 ) : #psi(left)=0, psi'(left)=0, i=j+2
    c0[2:basis_dim+2]=eigen_vec[0:basis_dim,iv].real # needs to be checked
    c1[2:basis_dim+2]=eigen_vec[0:basis_dim,iv].imag # needs to be checked
      #if no conditions on the right or psi'(left)=0,  only the last function is modified
  if(boundary_right==2) : # needs to be checked
    c0[nbasis_dim-1]=c0[basis_dim-2] # needs to be checked
    c1[nbasis_dim-1]=c1[basis_dim-2] # needs to be checked


  # plt.title("wave function, v="+ " {}".format(iv))

#  c0[0:n_inter+2*order] =0.; c[i]=1.
  npoints_for_plot=1000
# real part of the wave function
  spl = BSpline(t_knots, c0, order)
  x_new = np.linspace(min(t_knots), max(t_knots), npoints_for_plot)
  y_fit = BSpline(t_knots, c0, order)(x_new)
  # plt.xscale('log', base=10)
  # plt.plot(x_new, y_fit, '-r', label="v="+ " {}".format(iv)+" E="+ " {}".format(energies[iv]))

# imaginary part of the wave function
  spl = BSpline(t_knots, c1, order)
  y_fit2 = BSpline(t_knots, c1, order)(x_new)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
  ax1.set_title("wave function, v="+ " {}".format(iv))
  ax1.plot(x_new, y_fit, '-r', label="v="+ " {}".format(iv)+" E="+ " {}".format(energies[iv]))
    
  # FFT
  Yk = np.fft.fftshift(np.fft.fft(y_fit + 1j*y_fit2))
  k = np.fft.fftshift(np.fft.fftfreq(x_new.shape[-1], d=np.diff(x_new)[-1])) * 2*np.pi
  ax2.plot(k, Yk.real, 'r')
  ax2.plot(k, Yk.imag, 'b')

  ax1.legend(loc='best', fancybox=True, shadow=True)
  ax1.plot(x_new, y_fit2, '-b', label="imag")
  # plt.grid()
  ax1.grid()
  ax2.grid()
  plt.show() 
# ****************************************


# ****************************************
def sort_E_Bspl(NomE,DenomE,vr,N):
  energies=np.zeros(N,dtype =float)
  widths  =np.zeros(N,dtype =float)
  tmpv  =np.zeros(N,dtype =complex)
  for i in range(N):
    if( abs(DenomE[i]) > 1.e-10 ):
      z=NomE[i]/DenomE[i]
      energies[i]=z.real
      widths[i]=-2*z.imag
      iloc=maxloc(vr[:,i],N)
      tmp=abs(vr[iloc,i])/vr[iloc,i]
      vr[:,i]=vr[:,i]*tmp
# ordering 
  for i in range(N):
    for k in range(i+1,N):
      if ( energies[i] > energies[k] ):
        tmp =energies[i]; energies[i]=energies[k];energies[k]=tmp
        tmp =widths[i];   widths[i]  =widths[k];  widths[k]  =tmp
        tmpv=vr[:,i];     vr[:,i]    =vr[:,k];    vr[:,k]=tmpv
  eigenvectors=vr  
  return energies, widths, eigenvectors
   
# ****************************************
def maxloc(a,N):
  iloc=0
  for i in range(N):
    if(abs(a[i]) > abs(a[iloc]) ): iloc=i
  return iloc
# ****************************************



# Main code
# ****************************************

mass = 7294.0 
# n_inter=200

n_inter=500
x_begin=.05 *  1.8897e-5; x_end=400 *  1.8897e-5
# x_begin=.5; x_end=100 
# x_begin=.5; x_end=300 

# resonances = 0
resonances = 1

order=5
LegPoints=8
x_grid=np.zeros([n_inter,LegPoints],dtype =float)
x_weigths=np.zeros([n_inter,LegPoints],dtype =float)
x_sectors=np.zeros(n_inter+1)
x_grid,x_weigths,x_sectors=grid_for_1D_Bsplines(n_inter,LegPoints,x_begin,x_end)

t_knots=np.linspace(0., 0., n_inter+2*order+1)
t_knots[0:order] = x_begin
t_knots[n_inter+order+1:n_inter+2*order+1] = x_sectors[n_inter]

for i in range(n_inter+1):
  t_knots[order+i]=x_sectors[i]
#print (t_knots)

basis_dim = 0
boundary_left = 1; boundary_right = 1
basis_dim,bf =  constructing_basis(boundary_left,boundary_right)
#print ('checking basis',bf[5].f_on_grid)

fo =  open("basis2.dat", "w")
fod = open("basis_der2.dat", "w")

for i in range (basis_dim) :
  first_sec=bf[i].first_sector
  for k_sec in range(order+1):
    if(k_sec+first_sec<n_inter):
      for l in range (LegPoints) :
        tmp= "{:.4e}".format(x_grid[k_sec+first_sec,l])+ " {:.4e}".format(bf[i].f_on_grid[k_sec,l])
        tmpd="{:.4e}".format(x_grid[k_sec+first_sec,l])+ " {:.4e}".format(bf[i].f_on_grid_der[k_sec,l])
        fo.write(tmp+'\n')
        fod.write(tmpd+'\n')
  fo.write('\n')  
  fod.write('\n')  
fo.close()
fod.close()
# ****************************************



# plotting potential
plotting_pot()

if( resonances == 0):

  hamiltonian = np.zeros([basis_dim, basis_dim], dtype=float)
  overlap_mat = np.zeros([basis_dim, basis_dim], dtype=float)
  construct_hamiltonian(hamiltonian,overlap_mat)

  #diagonalization
  lwork=3*basis_dim-1;itype=1;jobz="V" ;uplo="L"
  energies,eigenvectors, info = lapack.dsygv(hamiltonian,overlap_mat,itype, jobz, uplo,lwork)
  if(info == 0):
    print('energies are')
    for i in range(100):
      print(i,energies[i])
#   plotting wave functions, obtained numericaly
    iv=0
    plotting_wf(eigenvectors,iv)
    iv=20
    plotting_wf(eigenvectors,iv)
    iv=33
    plotting_wf(eigenvectors,iv)
    iv=34
    plotting_wf(eigenvectors,iv)
  else:
    print('info=',info)


else :

  # complex-valued symmetric hamiltonian
  hamiltonian = np.zeros([basis_dim, basis_dim], dtype=complex)
  overlap_mat = np.zeros([basis_dim, basis_dim], dtype=complex)
  construct_hamiltonian(hamiltonian,overlap_mat)
  NomE,DenomE,vl,vr,work, info = lapack.zggev(hamiltonian,overlap_mat)
  if(info == 0):
    energies,widths,eigenvectors=sort_E_Bspl(NomE,DenomE,vr,basis_dim)

    n_E_show = 200
    0
    # n_E_show = energies.size


    print('energies and lifetimes are')
    for i in range(n_E_show):
      print(i,energies[i]/36749.7, (1./widths[i])*2.418884e-17)

    for iv in range(0,41):

      # if(widths[iv] < 1e-3):
      plotting_wf_c(eigenvectors,iv)

    print(f"N energies = {energies.shape}")
    plt.scatter(1/widths[:n_E_show] *2.418884e-17 , energies[:n_E_show]/36749.7, marker='+')
    plt.xscale('log', base=10)
    plt.grid()
    # plt.ylim(1.e-7, 1)
    # plt.xlim(-0.25, 1)
    plt.ylabel("energies, MeV")
    plt.xlabel("1/widths, s")
    plt.show()


    plotting_pot(energies[:n_E_show]/36749.7)

  else:
    print('info=',info)

#print('analytical infinite square-well energies')
#for i in range(1,16):
#  print(i,0.5/mass*(i*np.pi/10.)**2)







