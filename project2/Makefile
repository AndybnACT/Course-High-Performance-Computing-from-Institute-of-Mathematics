MPI_HOME = /opt/openmpi/1.6.3
MPI_RUN = /opt/openmpi/1.6.3/bin/mpiexec

CC = mpicc
CCFLAGS = -I${MPI_HOME}/include -c

pj2: mpi_square_matvec.o hpc_project2.o
	${CC} -o pj2 mpi_square_matvec.o hpc_project2.o
pj2j: mpi_square_matvec.o hpc_project2_j.o
	${CC} -o pj2j mpi_square_matvec.o hpc_project2_j.o
run:
	${MPI_RUN} -np ${p} ${target} 0.1 0.1

.PHONY: pj2 run pj2j

hpc_project2_j.o: hpc_project2_j.c mpi_square_matvec.h
	${CC} ${CCFLAGS} hpc_project2_j.c

mpi_square_matvec.o: mpi_square_matvec.c mpi_square_matvec.h
	${CC} ${CCFLAGS} mpi_square_matvec.c
hpc_project2.o: hpc_project2.c mpi_square_matvec.h
	${CC} ${CCFLAGS} hpc_project2.c
