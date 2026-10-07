program main
use M_initial
use M_evolve
use M_pars
use M_derivs	
use M_RHS


implicit none

real*8, dimension(:,:), allocatable :: Unew,Uold,du,ddu
real*8, dimension(:,:), allocatable :: iniu
real*8, dimension(:,:), allocatable :: Cnew	
real*8, dimension(:), allocatable :: x,Ztt, gammatt, deter,T
real*8 :: dx,dt,time
integer :: i,j,Nv,Nspatial,Ninigrid
integer :: gft_out_full,gft_out_brief,ret,gft_read_full
real*8 :: r
integer,dimension(3) :: iHor
integer :: count, count_rate, count_max, count_0, count_1
real*8 :: sigma, dxphi,pi
character*32 :: cname


!read pars
call readpars

!number of variables
Nv = 2
Nspatial = 2


Ninigrid = (N-1)*inigrid +1 
write(*,*) Ninigrid, N

!allocate needed memmory
!iniu has the variables used to solve the GR initail data
!Unew will have the updated values of the fields
!Uold will have the old values of the fields
!x are coords 

allocate(Unew(Nv,N),Uold(Nv,N),du(Nv,N),ddu(Nv,N),x(Ninigrid))
allocate(iniu(Nspatial,Ninigrid))
allocate(Cnew(8,N))
allocate(Ztt(N),gammatt(N), deter(N),T(N))

dx = (rout-rin)/(Ninigrid-1)
dt = cfl * abs(dx)

do i = 1, Ninigrid
	x(i) =  rin + (i-(1))*dx
end do

write(*,*) dx

! We call the initial data
call initialMOD(iniu,Uold,Cnew,dx,x,Nv,Nspatial,Ninigrid,iHor)


ret = gft_out_full('psi',time, ninigrid, 'x', 1, x, iniu(ipsi,:))

open (unit=15, file = 'Inidat.dat', status = 'replace')

do i=1,Ninigrid
	write(15,*) x(i), iniu(iphi,i), iniu(ipsi,i)
end do 
 
write(*,*) "Final psi=", iniu(ipsi,Ninigrid) , iniu(ipsi,(Ninigrid/3)*2)   
  
deallocate(x)
allocate(x(N))

dx = (rout-rin)/(N-1)
dt = cfl * abs(dx)

do i = 1, N
	x(i) =  rin + (i-(1))*dx
end do

write(*,*) dx

time = initime

!Depends on r_in
if(x(1).eq.0.0) then
iHor(:) = 0 
else
iHor(:) = 0
end if !!!WEll probably I dont need this guy

iHor(3) = Phor ! Number of points I want to keep behind the event horizon


      
do i=1,N
	Unew(:,i) = Uold(:,i)	
end do 

if(lev.ne.0) then
	
	ret = gft_read_full('phi',lev,n,cname,1,time,x,unew(iphi,:))
	ret = gft_read_full('sigma',lev,n,cname,1,time,x,unew(isigma,:))
	
	do i=1,N
	Uold(:,i) = Unew(:,i)  	
	end do 
	
	initime = time
			
end if 


pi = 4.0d0*atan(1.0d0)

do i=1,N 
	r = x(i)
	
	T(i) = Mass*1.0/2877.503896*r**2*(-exp(-(r-6.7)**2/sg**2)/(2.0*pi*sg)**(3./2.))	
			
end do




!!!!define if you want to use .sdf output
#define SDF

  
#ifdef SDF
if ( OutPutFields ) then
		
	ret = gft_out_full('sigma',time, n, 'x', 1, x, Uold(isigma,:))
	ret = gft_out_full('phi',time, n, 'x', 1, x, Uold(iphi,:))
	ret = gft_out_full('T',time, n, 'x', 1, x, T(:))	
end if

#endif	


!Prepearing some output  



write(*,*) time,unew(iphi,1)



!evolution 
do ITE=1, nt

 !call system_clock(count, count_rate, count_max)
  !count_0 = count         
	call evolve(unew,uold,dx,dt,time,x,Nv,N,iHor) !evolution routine

	time = initime + ITE*dt

	Uold = Unew
	
        do i=1,N
                if (isnan(unew(iphi,i))) then
                write(*,*) "boom"
                        stop
                end if
        end do

	if(mod(ITE,freqsdf).eq.0) then !output
!		do j=1, Nv
!			call derivs(unew(j,:),du(j,:),dx, N,1,N,j,iHor) 
!			call dderivs(unew(j,:),ddu(j,:),dx, N,1,N,j,iHor)

!		end do
		write(*,*) time
		
		
		call derivs(unew(iphi,:),du(iphi,:),dx,N,1,N,j,iHor)		
		call dderivs(unew(iphi,:),ddu(iphi,:),dx,N,1,N,j,iHor)
		
		
		do i=1,N
			dxphi = du(iphi,i)
			sigma = unew(isigma,i)
			
			deter(i) = (15*Sigma**4*cgamma-30*Sigma**2*cgamma*dxphi**2+15*cgamma*dxp&
     				&hi**4-12*Sigma**2*beta+12*beta*dxphi**2-4*csigma)/(3*Sigma**4*cgam&
     				&ma-6*Sigma**2*cgamma*dxphi**2+3*cgamma*dxphi**4-4*Sigma**2*beta+4*&
     				beta*dxphi**2-4*csigma)

     				
     			gammatt(i) = -(15*Sigma**4*cgamma-18*dxphi**2*Sigma**2*cgamma+3*dxphi**4*c&
     				&gamma-12*Sigma**2*beta+4*dxphi**2*beta-4*csigma)/(3*Sigma**4*cgamm&
     				&a-6*dxphi**2*Sigma**2*cgamma+3*dxphi**4*cgamma-4*Sigma**2*beta+4*d&
     				&xphi**2*beta-4*csigma)
	
		end do
		
		
		if ( OutPutFields ) then
#ifdef SDF
		
			ret = gft_out_full('sigma',time, n, 'x', 1, x, Unew(isigma,:))
			ret = gft_out_full('phi',time, n, 'x', 1, x, Unew(iphi,:))
			ret = gft_out_full('det',time, n, 'x', 1, x, deter(:))
			ret = gft_out_full('gammatt',time, n, 'x', 1, x, gammatt(:))												
#endif								
		end if
           
	end if
	
end do !end main do

end program main
