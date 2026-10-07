MODULE M_BOUNDARY	
implicit none	
contains
		
subroutine boundary(u,du,ddu,source,x,N,time)
use m_pars, only: isigma,iphi,csigma,beta,cgamma,Mpl,sg
implicit none	
real*8, dimension(:,:), intent(in) :: u,du,ddu
real*8, dimension(:), intent(in) :: x	
real*8, dimension(:,:), intent(inout) :: source
real*8 :: sigma,phi,d2xphi,T,pi,r,factor,time,Mass,cgamma_v
integer ::N

!set boundary values	

source(isigma,N) = -du(isigma,N) - u(isigma,N)/x(N)
pi = 4.0d0*atan(1.0d0)

                
if(x(1).eq.0.0) then    

	r = 0.0
	
	T = Mass*1.0/2877.503896*r**2*(-exp(-(r-6.7)**2/sg**2)/(2.0*pi*sg)**(3./2.))
	
	sigma = u(isigma,1)
	phi = u(iphi,1)
	d2xphi = ddu(iphi,1)

        cgamma_v =0.0
	
	source(iphi,1) = sigma
	source(isigma,1) = (9*Mpl*d2xphi*Sigma**4*cgamma_v-12*Mpl*d2xphi*Sigma**2*beta-12*&
     			&Mpl*d2xphi*csigma-2*T)/Mpl/(15*Sigma**4*cgamma_v-12*Sigma**2*beta-4*&
     			&csigma)

       
end if 
			
end subroutine boundary	
	
end module M_BOUNDARY
