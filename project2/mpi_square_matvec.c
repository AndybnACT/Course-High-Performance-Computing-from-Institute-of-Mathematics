#include "mpi_square_matvec.h"
// #include <omp.h>


void MatrixInit(double *A, int local_m, int n) {
    MPI_Scatter(A, local_m*n, MPI_DOUBLE, A, local_m*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
void VectInit(double *x, int n){
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void MulMatVect(double *A, double *x_in, double *x_out, int local_m, int n) {
    size_t i, j;
    double CValue;
    //#pragma omp parallel for private(i, j, CValue)
    for (i = 0; i < local_m; i++) {
        CValue = 0.0;
        // #pragma omp parallel for pravite(i,j,CValue)
        for (j = 0; j < n; j++) {
            //printf("%f + %fx%f  \n",CValue, A[i*n+j], x_in[j] );
            CValue += A[i*n+j] * x_in[j];
        }
        x_out[i] = CValue;
    }
}

void AddVectVect(double *x_1, double *x_2, int local_m) {
    size_t i;
    for (i = 0; i < local_m; i++) {
        x_1[i] += x_2[i];
    }
}


void MulResult(double *dest, double *src, int local_m){
    MPI_Allgather(src, local_m, MPI_DOUBLE, dest, local_m, MPI_DOUBLE, MPI_COMM_WORLD);
}
