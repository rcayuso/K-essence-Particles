MODULE M_RHS
implicit none
contains

subroutine rhs_time(u,du,ddu,source,N,x,time)
USE m_tempvars
USE m_pars, only : isigma,iphi,Mpl,csigma,beta,cgamma,sg,Mass
implicit none
real*8, dimension(:,:), intent(inout) :: u, du, ddu
real*8, dimension(:,:), intent(inout) :: source
real*8, dimension(:), intent(in) :: x
real*8 :: time,T,pi,factor,cgamma_v	
integer :: i,N,ret,gft_out_full,j
logical :: ltrace


pi = 4.0d0*atan(1.0d0)
source(:,:) = 0.0d0

do i=1, N
	r = x(i)
	phi = u(iphi,i)
	dxphi = du(iphi,i)
	d2xphi = ddu(iphi,i)
	sigma = u(isigma,i)
	dxsig = du(isigma,i)  
		
	T = Mass*1.0/2877.503896*r**2*(-exp(-(r-6.7)**2/sg**2)/(2.0*pi*sg)**(3./2.))	
		
	cgamma_v = cgamma*(0.5 * (1 + tanh(20.0 * (r**2 / 4.5**2. - 1))))
						
	source(iphi,i) = sigma
	source(isigma,i) = ((2*Mpl*d2xphi*Sigma**4*cgamma_v*r+16*Mpl*dxSig*dxphi*Sigma**3*c&
     			&gamma*r-12*Mpl*d2xphi*dxphi**2*Sigma**2*cgamma_v*r-16*Mpl*dxSig*dxph&
     			&i**3*Sigma*cgamma_v*r+10*Mpl*d2xphi*dxphi**4*cgamma_v*r+4*Mpl*dxphi*Si&
     			&gma**4*cgamma_v-8*Mpl*dxphi**3*Sigma**2*cgamma_v+4*Mpl*dxphi**5*cgamma_v&
     			&-2*Mpl*d2xphi*Sigma**2*beta*r-8*Mpl*dxSig*dxphi*Sigma*beta*r+6*Mpl&
     			&*d2xphi*dxphi**2*beta*r-4*Mpl*dxphi*Sigma**2*beta+4*Mpl*dxphi**3*b&
     			&eta-2*Mpl*d2xphi*csigma*r-4*Mpl*dxphi*csigma-T*r)/Mpl/r/(5*Sigma**&
     			&4*cgamma_v-6*dxphi**2*Sigma**2*cgamma_v+dxphi**4*cgamma_v-3*Sigma**2*bet&
     			&a+dxphi**2*beta-csigma)/2)
end do

end subroutine rhs_time

subroutine rhs_spaceMOD(uguess,source,xm,T)
USE m_pars, only : ipsi,iphi,beta,cgamma,csigma,Mpl

implicit none
real*8 :: xm,T
real*8, dimension(:) :: source
real*8, dimension(:) :: uguess
real*8 :: psi,r,phi,cgamma_v

psi = uguess(ipsi)
phi = uguess(iphi) 
	
r = xm
source = 0.0
		
!Regularization at origin    
if(xm.eq.0.0) then   

	source(ipsi) = -1.0/Mpl/csigma*T/6
    	source(iphi) =  0.0  
    	
    	write(*,*) source  	    		
else 
	
	!cgamma_v = cgamma*(1.-exp(-r**2))
	cgamma_v = cgamma*(0.5 * (1 + tanh(20.0 * (r**2 / 4.5**2. - 1))))
	
	
	source(ipsi) = (-(4*Mpl*psi**5*cgamma_v+4*Mpl*psi**3*beta-4*Mpl*psi*csigma-T*r)&
			&/Mpl/r/(5*cgamma_v*psi**4+3*beta*psi**2-csigma)/2)

	
	source(iphi) =  psi
	!if(xm.lt.0.1) then
    	!write(*,*) source
    	!end if      	     			
end if

       
end subroutine rhs_spaceMOD



end module M_RHS
