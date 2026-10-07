module M_EVOLVE
implicit none

contains

subroutine evolve(unew,uold,dx,dt,time,x,Nv,N,iHor)
use m_derivs
use m_rhs
use m_pars, only : isigma,iphi,disip,initime,ITE,Phor
implicit none

real*8, dimension(:,:), intent(inout):: unew
real*8, dimension(:,:), intent(inout):: uold
integer,dimension(:),intent(inout):: iHor
real*8, dimension(:):: x	
real*8 dx, dt, time
integer :: Nv,N
real*8 :: xm
real*8, dimension(:,:), allocatable :: tempdis,u1,u2,u3,u4,du,ddu
real*8, dimension(:,:), allocatable :: urk1,urk2,urk3,urk4
real*8,dimension(500) :: l
integer i,j,m

!allocate memory
allocate(tempdis(Nv,N),u1(Nv,N),u2(Nv,N),u3(Nv,N),u4(Nv,N),du(Nv,N),ddu(Nv,N),urk1(Nv,N),urk2(Nv,N),urk3(Nv,N),urk4(Nv,N))


! if when evolving the horizon moves inwards in coordinate radious, then change the number of points in grid evolved past event horizon   
if(iHor(1).lt.iHor(2)) then
	iHor(3) = iHor(3) + (iHor(1) -iHor(2))
end if

! if when evolving the horizon moves outwards (in coordinate radious) (and the number of points keept is smaller thant Phor) , then change the number of points in grid evolved past event horizon.     
if(iHor(1).gt.iHor(2).and.(iHor(3)+(iHor(1) -iHor(2))).le.Phor) then
	iHor(3) = iHor(3) + (iHor(1) -iHor(2))
end if
      
u1 = uold
u2 = uold
u3 = uold
u4 = uold

!march up 1st step rk.
!notice the rk evoln step is always the same, only changing
!by the size of the step, which we pass to that routine

call rk_timeup(uold,uold,u1,du,ddu,tempdis,urk1,x,dx,0.5*dt,Nv,N,time,iHor)    
!now do the 2nd RK step.
call rk_timeup(uold,u1,u2,du,ddu,tempdis,urk2,x,dx,0.5*dt,Nv,N,time+0.5*dt,iHor)
!now do the 3rd RK step.
call rk_timeup(uold,u2,u3,du,ddu,tempdis,urk3,x,dx,dt,Nv,N,time+dt,iHor)
!now do the 4th RK step.
call rk_timeup(uold,u3,u4,du,ddu,tempdis,urk4,x,dx,dt,Nv,N,time+dt,iHor)  

	!now combine the RKs
	do i= 1,Nv 	
		unew(i,1:N)=uold(i,1:N) &
		& + dt/6.*(urk1(i,1:N) + 2.*urk2(i,1:N) &
		& + 2.*urk3(i,1:N) + urk4(i,1:N)) 
	end do
	
	!if we have an horizon the set to 0 all points more than iHor(3) points away from the horizon to 0	    
	if(iHor(1).gt.0) then
		do i= 1,Nv 	
			unew(i,1:iHor(1)-iHor(3))=0.0d0       		      
		end do
	end if

deallocate(tempdis,u1,u2,u3,u4,du,ddu,urk1,urk2,urk3,urk4)
end subroutine evolve

! RK intermidiate step for the time advanceA
subroutine rk_timeup(uold,uprev,unew,du,ddu,tempdis,urk,x,dx,dt,Nv,N,time,iHor)
use m_derivs
use m_boundary
use m_rhs
use m_pars, only : disip,dissorder
implicit none
real*8, dimension(:,:) :: uold,uprev,unew,du,ddu,urk,tempdis
real*8, dimension(:) :: x
real*8 :: dx, dt, time
integer,dimension(:), intent(in) ::iHor
integer :: Nv,N
integer :: i,j,k

! We need dissipation for time-evolved fields
do j=1,Nv	 
	!call dissip(uprev(j,:),tempdis(j,:),dx, N,1,N) !USE THIS LINE FOR DISSIPATION dissip
	if (dissorder.eq.65) then
		call dissip65(uprev(j,:),tempdis(j,:),dx,N,j,iHor)
	end if
	if (dissorder.eq.63) then	
		call dissip63(uprev(j,:),tempdis(j,:),dx,N,j,iHor)   
	end if
	if (dissorder.eq.84) then	
		call dissip84(uprev(j,:),tempdis(j,:),dx,N,j,iHor)   
	end if	
	
	call derivs(uprev(j,:),du(j,:),dx,N,1,N,j,iHor)
	call dderivs(uprev(j,:),ddu(j,:),dx,N,1,N,j,iHor)

end do

 
call rhs_time(uprev,du,ddu,urk,N,x,time) 
call boundary(uprev,du,ddu,urk,x,N,time)

    
do j=1,Nv
	if(x(1).eq.0.0d0) then 
		tempdis(j,1:8) = 0.0d0 !In case you want to turn of the dissipation at r=0
	end if 
	!tempdis(j,N) = 0.0d0  !In case you want to turn of the dissipation at boundary	  		    		

end do
   
!urk = urk + sign(1.0,dx)*disip*tempdis  !USE THIS LINE FOR DISSIPATION dissip	
if (dissorder.eq.65) then
	urk = urk -disip/(64.0d0*dx)*tempdis
end if

if (dissorder.eq.63) then
	urk = urk +disip/(64.0d0*dx)*tempdis
end if

if (dissorder.eq.84) then
	urk = urk +disip/(128.0d0*dx)*tempdis
end if		
    
     
do j = 1,Nv
	do i = 1, N         
		unew(j,i) = uold(j,i) + dt * urk(j,i) 
	end do
end do 

  
end subroutine rk_timeup

!RK full march radially for initial data
subroutine rk_fullradialMOD(V,iniu,x,dx,Nspatial,N,T)
use m_derivs
use m_rhs
use m_pars, only : ipsi,iphi
implicit none

real*8, dimension(:,:) :: iniu
real*8, dimension(:) :: T
real*8, dimension(:) :: x
real*8, dimension(:) :: V 
real*8 :: dx
integer :: Nspatial,N
real*8 :: xm,del
real*8, dimension(Nspatial) :: k1,k2,k3,k4,u1,u2,u3,u4,uguess
real*8 ::um 
integer, dimension(3) :: iHor
integer :: i,j,I0,ll


!starting values
do i=1,Nspatial
	iniu(i,1) = V(i)
end do     
  	 
I0 = 2

do i=I0,N
	ll  = 0
	if(i.eq.2.or.i.eq.N) then
		ll = 1
	end if
	
	del = dx
	uguess = iniu(:,i-1)
	um = T(i-1)	
	xm = x(i-1)
	call rhs_spaceMOD(uguess,k1,x(i-1),um)
	u1 = iniu(:,i-1)+0.5*del*k1

	um  = ( 9./16.*(T(i-1)+T(i)) - 1./16.* (T(i-2+ll)+T(i+1-ll)) )      
	xm  = 0.5*(x(i-1)+x(i))
	call rhs_spaceMOD(u1,k2,xm,um)
	u2  = iniu(:,i-1)+0.5*del*k2
       
	um  = ( 9./16.*(T(i-1)+T(i)) - 1./16.* (T(i-2+ll)+T(i+1-ll)) )
	xm  = 0.5*(x(i-1)+x(i))
	call rhs_spaceMOD(u2,k3,xm,um)
	u3  = iniu(:,i-1)+del*k3

	um  = T(i)
	uguess = u3
	xm  = x(i)
	call rhs_spaceMOD(u3,k4,x(i),um)

	iniu(ipsi,i) = iniu(ipsi,i-1)+del/6.0*(k1(ipsi)+2.0*k2(ipsi)+2.0*k3(ipsi)+k4(ipsi))	
	iniu(iphi,i) = iniu(iphi,i-1)+del/6.*(k1(iphi)+2.*k2(iphi)+2.*k3(iphi)+k4(iphi))
	 	
end do
end subroutine rk_fullradialMOD

end module M_EVOLVE
 
