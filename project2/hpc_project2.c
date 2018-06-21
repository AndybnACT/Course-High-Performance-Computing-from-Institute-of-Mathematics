#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "mpi_square_matvec.h"
#define PI 3.14159265359f
#define ITER_MAX 1000
//#define DEBUGENV
#define PERF
//#include "iterative.h"


inline double exactsol(double x, double t){
    return exp(t)*sin(PI*x);
}

int main(int argc, char *argv[]) {
    double dx, dt;
    size_t i, j;
    // user inputed dx dt
    if (argc != 3) {
        printf("usage: %s <dx> <dt>\n",argv[0]);
        return 1;
    }else{
        dx = atof(argv[1]);
        dt = atof(argv[2]);
    }
    // printf("dx = %f, dt = %f\n", dx, dt );

    //
    // generate iterative matrix and starting vector
    //


    int NX = (int)(1.0/dx)+1;
    int NT = (int)(1.0/dt)+1;


    //---> mpi initialization
    int proc_cnt, rank_id;
    int m = NX-2;
    int local_m;
    double *x_cur, *x_next;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);

    local_m = (int)(m-1)/proc_cnt + 1;
    x_next = (double*) malloc(local_m*sizeof(double));

    //---> starting vector
    x_cur = (double*) malloc((2+local_m*proc_cnt)*sizeof(double));

    double xvalue=0.0;
    for (i = 0; i < NX; i++) {
        x_cur[i] = exactsol(xvalue,0.0);
        xvalue += dx;
    }
    //---> iteration matrix
    double *ITER, coef;
    coef = dt/(dx*dx);
    if (rank_id == 0) {
        ITER = (double*) malloc(local_m*m*local_m*m*sizeof(double));
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                if (i == j) ITER[i*m+j] = -2.0*coef;
                else if (j == i+1 || j == i-1) ITER[i*m+j] = coef;
                else ITER[i*m+j] = 0.0;
            }
        }
        #ifdef DEBUGENV
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                printf("%lf\t",ITER[i*m+j]);
            }
            printf("\n");
        }
        for (i = 0; i < NX; i++) {
            printf("x[%d]=%lf\n",i, x_cur[i]);
        }
        #endif
    }else{
        ITER = (double*) malloc(local_m*m*sizeof(double));
    }
    MatrixInit(ITER, local_m, m);


    //
    // do the iterations
    //

    //---> time marching begin
    double t = 1.0;
    double *x_inner = x_cur+1;
    double total_time=0.0;
    for (; t*dt <= 1.0; t=t+1.0) {
        double st, fi;
        // computations
        MPI_Barrier(MPI_COMM_WORLD);
        st = MPI_Wtime();
        MulMatVect(ITER, x_inner, x_next, local_m, m);
        AddVectVect(x_next, x_inner+local_m*rank_id, local_m);
        for (i = 0; i < local_m; i++) {
            x_next[i] += dt*(1+PI*PI)*exactsol((double)(i+local_m*rank_id)*dx, t*dt);
        }
        MulResult(x_inner, x_next, local_m); // get the result and apply to next step
        MPI_Barrier(MPI_COMM_WORLD);
        fi = MPI_Wtime();
        total_time += fi-st;


        #ifdef PERF
        if (rank_id == 0 && (t+1.0)*dt > 1.0) {
            double err=0.0;
            for (i = 1; i < NX-1; i++) {
                err += (x_cur[i]-exactsol((double)i*dx,t*dt))*(x_cur[i]-exactsol((double)i*dx,t*dt));
            }
            err=sqrt(err);
            printf("T:%lf 2norm error at 1 sec:%lf\n",total_time, err);
        }
        #endif


        // boundary conditions
        x_cur[0]    = exactsol(0.0, t*dt);
        x_cur[NX-1] = exactsol(1.0, t*dt);
        x_cur[1]    += coef*x_cur[0];
        x_cur[NX-2] += coef*x_cur[NX-1];
        #ifdef DEBUGENV
        if (rank_id == 0) {
            for (i = 0; i < NX; i++) {
                printf("%e\t-->x[%d]=%lf\t%lf\n", x_cur[i]-exactsol((double)i*dx,t*dt), i, x_cur[i],exactsol((double)i*dx,t*dt));
            }
        }
        #endif

    }

    // compare to the analytical solution
    //
    //



    // double *A, *x_in, *x_out;
    // int m, n, local_m, local_n;
    // m = 7;
    // n = 5;
    // local_m = (int)(m-1)/proc_cnt + 1;
    // local_n = (int)(n-1)/proc_cnt + 1;
    //
    // x_in = (double*) malloc(local_n*proc_cnt*sizeof(double));
    // x_out = (double*) malloc(local_m*sizeof(double));
    // if (rank_id == 0) {
    //     A = (double*) malloc(m*n*sizeof(double));
    //     size_t i,j;
    //     for (i = 0; i < m; i++) {
    //         for (j = 0; j < n; j++) {
    //             A[i*n+j] = i*n+j;
    //             printf("initial: %d   (%d %d)=%f\n", rank_id, i, j, A[i*n+j] );
    //         }
    //     }
    //     for (i = 0; i < n; i++){
    //         x_in[i] = (double) i;
    //         printf("initial: x[%d]=%f\n",i,x_in[i]);
    //     }
    // }else{
    //     A = (double *) malloc(local_m*n*sizeof(double));
    // }
    // VectInit(x_in,n);
    // MatrixInit(A, local_m, n);
    //
    // MulMatVect(A,x_in, x_out, local_m, n);
    // MulResult(x_in, x_out, local_m);
    //
    // if (rank_id == 0) {
    //     size_t i;
    //     for (i = 0; i < m; i++) {
    //         printf("x[%d]=%f\n",i, x_in[i]);
    //     }
    // }
    //





    MPI_Finalize();
    return 0;
}




// void MPI_JACOBI(double *A, double *sol, double *x0, int local_m, int n) {
//     static double *JACOBI_MAT = NULL, *xtmp, *b;
//     size_t i,j,k;
//     if (!JACOBI_MAT) {
//         // iteration matrix
//         double coef;
//         xtmp = (double*) malloc(local_m*sizeof(double));
//         b    = (double*) malloc(local_m*sizeof(double));
//         JACOBI_MAT = (double*) malloc(local_m*n*sizeof(double));
//         memcpy(JACOBI_MAT, A, local_m*n*sizeof(double));
//         memcpy(b, sol, local_m*sizeof(double));
//         for (i = 0; i < local_m; i++) {
//             coef = 1.0/A[i*n+i];
//             b[i] *= coef;
//             JACOBI_MAT[i*n+i] = 0.0;
//             for (j = 0; j < n; j++) {
//                 JACOBI_MAT[i*n+j] *= coef;
//             }
//         }
//     }
//     // iteration
//     for (k = 0; k < ITER_MAX; k++) {
//         // compute nex x
//         MulMatVect(JACOBI_MAT, x0, xtmp, local_m, n);
//         // b - x
//         for (i = 0; i < local_m; i++) {
//             xtmp[i] = b[i] - xtmp[i];
//         }
//         // apply to new step
//         MulResult(x0, xtmp, local_m);
//         // calculate residual
//         MulMatVect(A, x0, xtmp, local_m, n);
//         double local_max = -1.0;
//         for (i = 0; i < local_m; i++){
//             xtmp[i] = fabs(sol[i] - xtmp[i]);
//             if (local_max < xtmp[i]) local_max = xtmp[i];
//         }
//         MPI_Allreduce(&local_max, &local_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//         if (local_max < 1e-4) {
//             break;
//         }
//
//
//     }
//
// }
