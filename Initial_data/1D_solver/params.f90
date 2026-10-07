MODULE M_PARS
implicit none

!define parameters !!!!!!!!!!!!!!!!!!!!!!!
real*8 ::  rin,rout,cfl,disip,kappa,amp,ct,sg,initime,Mass
real*8 :: tau,csigma,cgamma,beta,Mpl
integer :: N,Nt,freqsdf,freqhor,derorder,dderorder,tderorder,qderorder,dissorder,PHor,ITE
integer :: iphi,isigma,irho,ibeta,ipsi
integer :: inigrid
logical :: constr,Kretch,OutPutFields,nofix,tracking
integer :: lev

CONTAINS

!these are some generic parameters to use when the time calls for them	
subroutine readpars
implicit none
namelist /pars_input/ N,Nt,freqsdf,freqhor,PHor,cfl,disip, &
	&		derorder,dderorder,tderorder,qderorder,dissorder, &
	&		kappa,amp,ct,sg,initime,rin,rout,constr,Kretch,OutPutFields,nofix,&
	&              tau,Mass,tracking,Mpl,csigma,cgamma,beta,lev,inigrid
		
!read params !!!!!!!!!!!!!!!!!!!!!!!
open (unit = 10, file = "pars.in", status = "old" )
read (unit = 10, nml = pars_input)
 close(unit = 10)
     
! Output some parameters
print *, "Domain, rin =", rin, " to rout =", rout

!we add also some pars to label fields so that
!we can later call the fields without having to remember
!what index corresponds to what field

!Initial dara fields

ipsi = 2
iphi = 1
!iK = 3
    	
! Fields that evolve in time

!igrr = 1
!igT = 2 
!iKrr = 3
!iKT = 4
!ialp = 5
!ibeta = 6
!ipip = 7
!irho = 8
isigma = 2 
iphi = 1
irho=3
ibeta=4
!iPiKrr = 11
!iPiKT = 12
!iPipip = 13
!iPirho = 14


end subroutine readpars

end module M_pars
