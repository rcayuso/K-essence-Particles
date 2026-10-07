MODULE M_NEWTON_RAP
implicit none
contains

subroutine newt(V,nc,iniu,dx,N,x,Nv,Nspatial,tolf,tolx,T)
implicit none

real*8, dimension(:,:), intent(in) :: iniu
real*8, dimension(:), intent(in) :: x, T
real*8 :: dx			
integer :: nc, Nv, Nspatial, np
real*8, dimension(:), intent(inout) :: V
real*8, dimension(nc) :: Fvec
real*8 :: TOLF, TOLX, errf, errx
integer :: i, j, k, N
real*8, dimension(nc,nc) :: Fjac , Fjacin 
real*8, dimension(nc) :: p
real*8 :: detjac 
integer :: ntrial
	
ntrial = 100
     
do k = 1, ntrial
	write(*,*) "Newton-rapson iteration n", k ,V(1),fvec(1),V(2),fvec(2)

	call funcv(nc,V,fvec,iniu,x,dx,Nv,Nspatial,N,T)
	

	errf = 0.0
	do i = 1, nc
		errf = errf + abs(fvec(i))
	end do 

	if (errf.le.tolf) then
	write(*,*) "found a solution after interation", k, "with errors errf =", errf, "with errors errx =", errx
        	return
	end if
	
	call fdjac(nc,V,fvec,nc,fjac,iniu,x,dx,Nv,Nspatial,N,T)
	



!!!! WE need a solution for the equation J.dv = -F, which can be solved by getting dv = J^-1*(-F) !!!!
!! So lets just first invert J, since I am in a 2x2 this is in priciple easy but I might have to change this if I decide to include Krr in the problem, which I will probably have to.

	if(nc.eq.1) then
		!p(1) = fvec(1)/Fjac(1,1)
		p(1) = -fvec(1)/Fjac(1,1)
	else 
        
		detjac = Fjac(2,2)*Fjac(1,1) - Fjac(2,1)*Fjac(1,2)

		Fjacin(1,1) =  Fjac(2,2)/detjac
		Fjacin(2,2) =  Fjac(1,1)/detjac
		Fjacin(1,2) =  -Fjac(1,2)/detjac
		Fjacin(2,1) =  -Fjac(2,1)/detjac

		p(1) = -(Fjacin(1,1)*fvec(1) + Fjacin(1,2)*fvec(2)) ! solution 

		p(2) = -(Fjacin(2,1)*fvec(1) + Fjacin(2,2)*fvec(2)) ! solution
	end if

	errx = 0.0

	do i = 1, nc
		errx = errx + abs(p(i))
		V(i) = V(i) + p(i)
	end do

	write(*,*) errf,errx, "prposed steps", p 
	if(errx.le.tolx) then
		return
	end if
  
end do

end subroutine newt

 
subroutine fdjac(nc,V,fvec,NP,df,iniu,x,dx,Nv,Nspatial,N,T)
implicit none
  
integer :: nc , np, Nv, Nspatial
integer :: NMAX
real*8 , parameter :: Mpress = 1.e-4  
real*8, dimension(np,np) :: df
real*8, dimension(nc) :: fvec 
real*8, dimension(:) ::  V
integer :: i ,j
real*8 :: h , temp
real*8, dimension(nc) :: f
real*8, dimension(:,:), intent(in) :: iniu
real*8, dimension(:), intent(in) :: x, T
real*8 :: dx
integer :: N
NMAX = nc 			


do j= 1,nc
 
	temp = V(j)
	h = Mpress*abs(temp)
	
	if(h.eq.0.0) then
		h = Mpress
	end if
	 
	V(j) = temp + h     !!!! aparently to reduce finite precision error 
	h = V(j) - temp
	 
	call funcv(nc,V,f,iniu,x,dx,Nv,Nspatial,N,T)
		

	V(j) = temp ! we go back to the original V   
     
	do i = 1, nc
		df(i, j) = (f(i)-fvec(i))/h   !! computes the jacobian components	 
	end do
	
end do

  
end subroutine fdjac  

subroutine funcv(nc,V,fvec,iniu,x,dx,Nv,Nspatial,N,T)
use m_pars, only : ipsi,iphi
use M_EVOLVE

implicit none 
real*8, dimension(:), intent(in) :: V, T
real*8, dimension(:), intent(out) :: fvec
real*8, dimension(:), allocatable :: BC
integer :: nc, Nv, Nspatial
integer :: i
real*8, dimension(:,:), intent(in) :: iniu
real*8, dimension(:), intent(in) :: x
real*8 :: dx
integer :: N
 
allocate(BC(nc))
 
call rk_fullradialMOD(V,iniu,x,dx,Nspatial,N,T)

!BC(1) = iniu(iphi,N) !- x(N)*iniu(ipsi,N)
BC(1) = iniu(iphi,N) + x(N)*iniu(ipsi,N)
!BC(1) = iniu(iphi,N) - x(N)*iniu(ipsi,N)
fvec(1) = BC(1) 	
	
deallocate(BC)
end subroutine funcv

End module M_NEWTON_RAP 
