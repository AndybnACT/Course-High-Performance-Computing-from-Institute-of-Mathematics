#ifndef MPIMATVECT
#define MPIMATVECT
    #include "mpi.h"
    void MatrixInit(double *, int , int);
    void VectInit(double *, int);
    void MulMatVect(double *, double *, double *, int, int);
    void AddVectVect(double *, double *, int);
    void MulResult(double *, double *, int);
#endif
