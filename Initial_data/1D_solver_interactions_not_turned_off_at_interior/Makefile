#! /usr/sbin/smake
#
FC      = gfortran 
F90     = gfortran
CC      = ifc
LD      = $(F90)
MOD     = mod
SWP     =
RM      = /bin/rm -f
MP      =
ABI     =
ISA     =
ARCH    = 
OLEVEL  = #-O3
FOPTS   = #-O3
F90OPTS = -O3  # -H aesu
COPTS   =
LIBS    = -lbbhutil -lrnpl 
LDFLAGS = 
LIBPATH = -L/usr/local/intel81/lib/
F90FLAGS= $(ARCH) $(OLEVEL) $(F90OPTS) -ffree-line-length-180 -fdefault-real-8 -cpp #-ffpe-trap=overflow  -O -Wall -fcheck=all -g -fbacktrace
FFLAGS  = $(ARCH) $(OLEVEL) $(FOPTS)
CFLAGS  = $(ARCH) $(OLEVEL) $(COPTS)
LIBFLAGS = $(LIBPATH) $(LIBS)
PROF    =

PROG =	wave

SRCS =	boundary.f90 derivs.f90 evolve.f90  initial.f90 main.f90 rhs.f90 params.f90 tempvars.f90 Newton-rapson.f90
OBJS =	boundary.o derivs.o evolve.o  initial.o main.o rhs.o params.o tempvars.o Newton-rapson.o

all: $(PROG)

$(PROG): $(OBJS)
	$(F90) $(LDFLAGS) -o $@ $(OBJS) $(LIBFLAGS)

clean:
	rm -f $(PROG) $(OBJS) *.$(MOD)

tar:
	tar cf `basename $(PWD)`.tar $(SRCS) *.in makemake makefile makefile.* Makefile Config lib Backup Batch Inputs
	gzip -f `basename $(PWD)`.tar

.SUFFIXES: $(SUFFIXES) .f90

.f90.o:
	$(F90) $(F90FLAGS) -c $<

boundary.o:params.o
initial.o: Newton-rapson.o
evolve.o: boundary.o derivs.o rhs.o
main.o: params.o evolve.o initial.o
rhs.o: tempvars.o 
