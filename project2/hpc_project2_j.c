#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "mpi_square_matvec.h"
#define PI 3.14159265359f
#define ITER_MAX 1000
// #define DEBUGENV
#define DEBUGSOL
#define PERF
//#include "iterative.h"


inline double exactsol(double x, double t){
    return exp(t)*sin(PI*x);
}
int MPI_JACOBI(double *A, double *b, double *x0, int local_m, int n, int rank_id) {
    static double *JACOBI_MAT = NULL, *xtmp, coef, *coef_list;
    size_t i,j,k;
    if (!A) {
        free(JACOBI_MAT);
        JACOBI_MAT = NULL;
        free(coef_list);
        coef_list = NULL;
        free(xtmp);
        xtmp = NULL;
    }
    if (!JACOBI_MAT) {// first invoked
        // iteration matrix
        xtmp = (double*) malloc(local_m*sizeof(double));
        coef_list = (double*) malloc(local_m*sizeof(double));
        JACOBI_MAT = (double*) malloc(local_m*n*sizeof(double));
        memcpy(JACOBI_MAT, A, local_m*n*sizeof(double));
        // memcpy(b, sol, local_m*sizeof(double));
        for (i = 0; i < local_m; i++) {
            size_t Matindex = i*n+i+rank_id*local_m;
            if (Matindex >= local_m*n) {   // 0 x x
                coef_list[i] = 0.0;        // 0 0 0 i<--Matindex
                b[i] = 0.0;
                continue;
            }else if (A[Matindex] == 0) { // x x 0
                coef_list[i] = 0.0;       // 0 0 0<-- Matindex
                b[i] = 0.0;
                continue;
            }
            coef = 1.0/A[Matindex];
            coef_list[i] = coef;
            b[i] *= coef;
            JACOBI_MAT[Matindex] = 0.0;
            for (j = 0; j < n; j++) {
                JACOBI_MAT[i*n+j] *= coef;
            }
        }
        #ifdef DEBUGENV
        for (i = 0; i < local_m; i++) {
            for (j = 0; j < n; j++) {
                printf("%lf\t",JACOBI_MAT[i*n+j]);
            }
            printf("<---%lf==>%d\n",b[i],rank_id);
        }
        #endif
    }else{
        for (i = 0; i < local_m; i++) b[i] *= coef_list[i];
    }



    // iteration
    double local_max;
    for (k = 0; k < ITER_MAX; k++) {
        local_max = -1.0;
        // compute nex x
        MulMatVect(JACOBI_MAT, x0, xtmp, local_m, n);
        // b - x
        for (i = 0; i < local_m; i++) {
            xtmp[i] = b[i] - xtmp[i];
        }
        // apply to new step
        MulResult(x0, xtmp, local_m);
        // calculate residual
        MulMatVect(A, x0, xtmp, local_m, n);

        for (i = 0; i < local_m; i++){
            if (coef_list[i] == 0) continue;
            xtmp[i] = fabs(b[i]/coef_list[i] - xtmp[i]); //it's not very ideal to do '/' here
            if (local_max < xtmp[i]) local_max = xtmp[i];
        }
        MPI_Allreduce(&local_max, &local_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (local_max < 1e-4) {
            return k;
        }

    }
    #ifdef DEBUGENV
    printf("WARNING--->REACHED MAXIMUM ITERATIONS r=%lf\n", local_max);
    #endif
    return ITER_MAX;

}

int main(int argc, char *argv[]) {
    double total_time = 0.0;
    double avg_err = 0.0;
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
    int NX = (int)(1.0/dx)+1;
    int NT = (int)(1.0/dt)+1;

    // printf("dx = %f, dt = %f\n", dx, dt );

    //
    // generate iterative matrix and starting vector
    //

    //---> mpi initialization
    int proc_cnt, rank_id;
    int m = NX-2;
    int local_m;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);

    local_m = (int)(m-1)/proc_cnt + 1;


    //---> starting vector

    double *x_cur, *x_next;
    x_cur = (double*) calloc(local_m*proc_cnt+2,sizeof(double));

    double xvalue=0.0;
    for (i = 0; i < NX; i++) {
        x_cur[i] = exactsol(xvalue,0.0);
        xvalue += dx;
    }


    x_next = (double*) calloc(local_m*proc_cnt,sizeof(double));
    //printf("m=%d  local_m=%d\n",m,local_m );
    //---> iteration matrix
    double *ITER, coef;
    coef = dt/(dx*dx);
    if (rank_id == 0) {
        // printf("local_m*proc_cnt=%d, m=%d\n", local_m*proc_cnt, m);
        ITER = (double*) calloc((local_m*proc_cnt)*(local_m*proc_cnt),sizeof(double));
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                if (i == j) ITER[i*m+j] = 1.0 + 2.0*coef;
                else if (j == i+1 || j == i-1) ITER[i*m+j] = -coef;
                //else ITER[i*m+j] = 0.0;
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

    for (; t*dt <= 1.0; t=t+1.0) {
        int iter;
        // boundary conditions
        x_cur[0]    = exactsol(0.0, t*dt);
        x_cur[NX-1] = exactsol(1.0, t*dt);
        x_cur[1]    += coef*x_cur[0];
        x_cur[NX-2] += coef*x_cur[NX-1];

        double *rhs = x_inner+local_m*rank_id;
        for (i = 0; i < local_m; i++) {
            rhs[i] += dt*(1+PI*PI)*exactsol((double)(i+local_m*rank_id)*dx, (t+1.0)*dt);
        }
        // solve the linear system
        double st,fi;
        MPI_Barrier(MPI_COMM_WORLD);
        st = MPI_Wtime();
        iter = MPI_JACOBI(ITER, rhs, x_next, local_m, m, rank_id);
        MPI_Barrier(MPI_COMM_WORLD);
        fi = MPI_Wtime();
        total_time += fi-st;
        // apply to the next step
        memcpy(x_inner, x_next, m*sizeof(double));



        #ifdef DEBUGSOL
        if (rank_id == 0) {
            double err=0.0, errmax_at_tstep = 0.0;
            errmax_at_tstep = 0.0;
            printf("== absolute err ==== numerical sol ==== analytical sol ================\n");
            for (i = 0; i < NX; i++) {
                err = x_cur[i]-exactsol((double)i*dx,t*dt);
                printf("%e\t-->x[%d]=%lf\t%lf\n", err, i, x_cur[i],exactsol((double)i*dx,t*dt));
                if (errmax_at_tstep < err) {
                    errmax_at_tstep = err;
                }
            }
            printf("MPI JACOBI ITERATION COUNT = %d, ERRMAX=%lf\n", iter, errmax_at_tstep);
            printf("[%d]th step, Time spent=%lf, processes count=%d\n",(int)t,fi-st,proc_cnt );

        }
        #endif

        #ifdef PERF
        if (rank_id == 0 && (t+1.0)*dt >1.0) {
            double err=0.0;
            for (i = 1; i < NX-1; i++) {
                err += (x_cur[i]-exactsol((double)i*dx,t*dt))*(x_cur[i]-exactsol((double)i*dx,t*dt));
            }
            err=sqrt(err);
            printf("T:%lf 2norm error at 1 sec:%lf\n",total_time, err);
        }
        #endif

    }

    free(ITER);
    ITER = NULL;
    free(x_cur);
    x_cur = NULL;
    free(x_next);
    x_next = NULL;
    MPI_JACOBI(NULL,NULL,NULL,0,0,0);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
