MODULE M_INITIAL
use M_pars
use M_NEWTON_RAP
use M_EVOLVE
implicit none
CONTAINS

subroutine initialMOD(iniU,Uold,Cnew,dx,x,Nv,Nspatial,N,iHor)
use m_pars, only : isigma,iphi,sg,inigrid,Mass
implicit none

real*8, dimension(:,:) :: Uold,Cnew 
real*8, dimension(:,:) :: iniU
real*8, dimension(:) :: x
real*8 :: dx, dy,dt
integer :: Nv,Nspatial, N, i, j ,m, ret, gft_out_brief,nc
integer :: Nevol 
integer,dimension(:),intent(in)::iHor
real*8 :: pi, tolf, tolx,r
real*8, dimension(:), allocatable :: T
real*8, dimension(:), allocatable :: V
allocate(T(N),V(2))

pi = 4.0d0*atan(1.0d0)
Nevol = (N-1)/inigrid + 1
do i=1,N 
	r = x(i)

	T(i) = Mass*1.0/2877.503896*r**2*(-exp(-(r-6.7)**2/sg**2)/(2.0*pi*sg)**(3./2.))
	

end do


V(ipsi) = 0.0 
V(iphi) = 0.0

tolf = 9e-18 !some tolerance for the newton-rapson
tolx = 9e-18 !some tolerance for the newton-rapson
nc = 1 
!Newton-Rapson
call newt(V,nc,iniu,dx,N,x,Nv,Nspatial,tolf,tolx,T)

!call rk_fullradialMOD(V,iniu,x,dx,Nspatial,N,T)

!iniU(:,:) = 0.0d0

!Cnew(:,:) = 0.0d0
 
do i=1,Nevol

	uold(iphi,i) = iniu(iphi,(i-1)*inigrid + 1) 
	uold(isigma,i) = 0.0	
end do


end subroutine initialMOD

END MODULE M_INITIAL
