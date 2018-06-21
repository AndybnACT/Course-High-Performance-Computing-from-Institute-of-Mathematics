#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
// #define DEBUG
#define DEBUGSOL
#define JCBDEBUG
#define GSDEBUG
#define SORDEBUG
#define float double
#ifdef DEBUGSOL
    float *EXACT;
    float absolute_err(float *, float *, size_t);
#endif


#define PI 3.14159265359f
#define ID(row,col,n) row*n+col // row-major indecing
#define ITER_MAX 5000
#define ERRCRE 1e-6

int nineptr_mat(float **mat, int n, float lim){
    // generate 9-point centered FD stencils for Laplacian operator
    // called by main
    int nn = n*n;
    float hcoef, step;
    // Any row or col should contain all grid points
    // -->eg: (0,0) (0,1) (0,2) (1,0) (1,1) (1,2) (2,0) (2,1) (2,2)
    step  = (lim/(n+1));
    hcoef = 1.0/(12.0*step*step);
    //hcoef = 1;
    *mat = (float *) malloc(n*n*n*n*sizeof(float));
    if (!(*mat)) {
        perror("Error insufficient memory");
        return -1;
    }
    for (size_t i = 0; i < nn; i++) {
        for (size_t j = 0; j < nn; j++) {
            if (i == j) (*mat)[ID(i,j,nn)] = -8.0*hcoef;
            else if (j == i-n || j == i+n) (*mat)[ID(i,j,nn)] = hcoef;
            // grid points at the left boundary shall not get values from its left
            // , or it would get the last velue from previous row due to the memory layout of our matriies
            else if (i%n != 0 && (j == i-1 || j == i-n-1 || j == i+n-1) ) (*mat)[ID(i,j,nn)] = hcoef;
            // grid points at the right boundary shall not get values from its right
            else if (i%n != n-1 && (j == i+1 || j == i+n+1 || j == i-n+1) ) (*mat)[ID(i,j,nn)] = hcoef;
            else (*mat)[ID(i,j,nn)] = 0.0;
        }
    }
    return 0;
}
static inline float exactsol(float x, float y){
    // exact solution to the problem
    // called by NorthBCAply to apply boundary value
    float coef=1.0/(exp(-PI)-exp(PI));
    return coef*sin(PI*x)*(exp(-PI*y)-exp(PI*y));
}
void NorthBCAply(float *b, int n, float (*fun)(float,float), char bdtype, float lim){
    float bc[3];
    float step = lim/(n+1);
    float hcoef = 1.0/(12.0*step*step);
    //hcoef = 1.0;
    bc[0] = fun(0,lim)*hcoef;
    bc[1] = fun(step,lim)*hcoef;
    bc[2] = fun(step*2,lim)*hcoef;

    size_t base = n*n-n;
    for (size_t i = 0; i < n; i++) {
        b[base+i] = -(bc[0]+bc[1]+bc[2]);
        bc[i%3] = fun((i+3)*step,lim)*hcoef;
    }
    #ifdef DEBUG
    for (size_t i = 0; i < n+2; i++) {
        printf("x=%f north bc=%f\n",i*step, fun(i*step,lim));
    }
    for (size_t i = 0; i < n*n; i++) {
        printf("%f\n",b[i] );
    }
    #endif
}

float Iterate_Jocobi(float *A, float *b, float **u, float **u_0, size_t nn){
    // Variable will be written after this function:
    //         A->scaled (diag(A) won't be removed since we want it for calculating residual)
    //         b->scaled
    //         u->solution
    //       u_0->residual vector
    // Called by main
    // Variables:
    //             u_0--> initial guess
    //         A u = b--> given value based on BC
    //         | |------> final solution
    //         |--------> stencil matrix
    // Jocobi method, Q=diag(A)
    // x_(k+1) = (I-inv(Q)*A)*x_(k) + inv(Q)b
    //                   |b|   | 0 .  . | |x_k|
    //         = 1/a_ii( |b| - | . 0  . |*|x_k| )
    //                   |b|   | . .  0 | |x_k|
    float coef, cvalue;
    float *tmp;
    // preparations
    for (size_t i = 0; i < nn; i++) {
        coef = 1.0/A[ID(i,i,nn)];
        b[i] *= coef;
        for (size_t j = 0; j < nn; j++) {
            A[ID(i,j,nn)] *= coef;
        }
    }
    // iteration
    for (size_t k = 0; k < ITER_MAX; k++) {
    //--> computing new x
        for (size_t i = 0; i < nn; i++) {
            cvalue = 0.0;
            for (size_t j = 0; j < nn; j++) {
                if (i!=j) cvalue += A[ID(i,j,nn)]*(*u_0)[j];
            }
            (*u)[i] = b[i] - cvalue;
        }
    //--> checking error, here we use u_0 to store residual
        for (size_t i = 0; i < nn; i++) {
            cvalue = 0.0;
            for (size_t j = 0; j < nn; j++) {
                cvalue += A[ID(i,j,nn)]*(*u)[j];
            }
            (*u_0)[i] = fabs(cvalue-b[i]);
        }
        // use inf norm
        cvalue = 0.0;
        for (size_t i = 0; i < nn; i++) {
            if (cvalue <= (*u_0)[i]) cvalue = (*u_0)[i];
        }
        #ifdef JCBDEBUG
            printf("%zu: %e\t",k+1, cvalue);
            #ifdef DEBUGSOL
                printf("%e ", absolute_err(*u,EXACT,nn));
            #endif
            printf("\n");
        #endif
        if (cvalue <= ERRCRE) return cvalue;
    //--> applying u to new initial guess;
        tmp  = *u_0;
        *u_0 = *u;
        *u   = tmp;
    }
    printf("[ERROR]Jocobi method can not converge to the specified cretiria %e\n",ERRCRE);
    tmp  = *u_0;
    *u_0 = *u;
    *u   = tmp;
    return cvalue;
}

float Iterate_GS(float *A, float *b, float *u, float *u_0, size_t nn){
    // Variable will be written after this function:
    //         A->scaled (diag(A) won't be removed since we want it for calculating residual)
    //         b->scaled
    //         u->solution
    //       u_0->residual vector
    // Called by main
    // Variables:
    //             u_0--> initial guess
    //         A u = b--> given value based on BC
    //         | |------> final solution
    //         |--------> stencil matrix
    // Gauss-Sedial method, Q=diag(A), while update x once it's ready
    // (apply x immediately)
    // x_(k+1) = (I-inv(Q)*A)*x_(k) + inv(Q)b
    //                   |b|   | 0 .  . | |x_k|
    //         = 1/a_ii( |b| - | . 0  . |*|x_k| )
    //                   |b|   | . .  0 | |x_k|
    float coef, cvalue;
    float *tmp;
    // preparations
    for (size_t i = 0; i < nn; i++) {
        coef = 1.0/A[ID(i,i,nn)];
        b[i] *= coef;
        for (size_t j = 0; j < nn; j++) {
            A[ID(i,j,nn)] *= coef;
        }
    }
    for (size_t i = 0; i < nn; i++) u[i] = u_0[i];
    // iteration
    for (size_t k = 0; k < ITER_MAX; k++) {
    //--> computing new x
        for (size_t i = 0; i < nn; i++) {
            cvalue = 0.0;
            for (size_t j = 0; j < nn; j++) {
                if (i!=j) cvalue += A[ID(i,j,nn)]*u[j];
            }
            u[i] = b[i] - cvalue;
        }
    //--> checking error, here we use u_0 to store residual
        for (size_t i = 0; i < nn; i++) {
            cvalue = 0.0;
            for (size_t j = 0; j < nn; j++) {
                cvalue += A[ID(i,j,nn)]*u[j];
            }
            u_0[i] = fabs(cvalue-b[i]);
        }
        // use inf norm
        cvalue = 0.0;
        for (size_t i = 0; i < nn; i++) {
            if (cvalue <= u_0[i]) cvalue = u_0[i];
        }
        #ifdef GSDEBUG
            printf("%zu: %e\t",k+1, cvalue);
            #ifdef DEBUGSOL
                printf("%e ", absolute_err(u,EXACT,nn));
            #endif
            printf("\n");
        #endif
        if (cvalue <= ERRCRE) return cvalue;
    }
    printf("[ERROR]Gauss-Sedial method can not converge to the specified cretiria %e\n",ERRCRE);
    return cvalue;
}

float Iterate_SOR(float *A, float *b, float **u, float **u_0, size_t nn, float w){
    // Variable will be written after this function:
    //         A->scaled
    //         b->scaled
    //         u->solution
    //       u_0->residual vector
    // Called by main
    // Variables:
    //             u_0--> initial guess
    //         A u = b--> given value based on BC
    //         | |------> final solution
    //         |--------> stencil matrix
    // SOR method, combination of GS and Jocobi
    //
    // x_(k+1) = inv(D+w*L)*(w*b - (w-1)*D*x_(k))
    //                                      w          n                   n
    // --------> xi_(k+1) = (1-w)*xi_(k) + --- ( bi - SUM(aij*xj_(k+1)) - SUM(aij*xj_(k)) )
    //                                     aii        j<i                 j>i
    float coef, cvalue, coef1mw;
    float *tmp;
    // preparations
    for (size_t i = 0; i < nn; i++) {
        coef = w/A[ID(i,i,nn)];
        b[i] *= coef;
        for (size_t j = 0; j < nn; j++) {
            A[ID(i,j,nn)] *= coef;
        }
    }
    for (size_t i = 0; i < nn; i++) (*u)[i] = (*u_0)[i];
    // iteration
    coef1mw = 1-w;
    for (size_t k = 0; k < ITER_MAX; k++) {
    //--> apply result from previous step to new initial guess
        //for (size_t i = 0; i < nn; i++) u_0[i] = u[i];
    //--> computing new x
        for (size_t i = 0; i < nn; i++) {
            cvalue = 0.0;
            for (size_t j = 0; j < i; j++) {
                cvalue += A[ID(i,j,nn)]*(*u)[j];
            }
            for (size_t j = i+1; j < nn; j++) {
                cvalue += A[ID(i,j,nn)]*(*u_0)[j];
            }
            (*u)[i] = coef1mw*(*u_0)[i] + b[i] - cvalue;
        }
    //--> checking error, here we use u_0 to store residual
        for (size_t i = 0; i < nn; i++) {
            cvalue = 0.0;
            for (size_t j = 0; j < nn; j++) {
                cvalue += A[ID(i,j,nn)]*(*u)[j];
            }
            (*u_0)[i] = fabs(cvalue-b[i]);
        }
        // use inf norm
        cvalue = 0.0;
        for (size_t i = 0; i < nn; i++) {
            if (cvalue <= (*u_0)[i]) cvalue = (*u_0)[i];
        }
        #ifdef SORDEBUG
            printf("%zu: %e\t",k+1, cvalue);
            #ifdef DEBUGSOL
                printf("%e ", absolute_err(*u,EXACT,nn));
            #endif
            printf("\n");
        #endif
        if (cvalue <= ERRCRE) return cvalue;
        tmp  = *u_0;
        *u_0 = *u;
        *u   = tmp;
    }
    printf("[ERROR]SOR method can not converge to the specified cretiria %e\n",ERRCRE);
    tmp  = *u_0;
    *u_0 = *u;
    *u   = tmp;
    return cvalue;
}

int main(int argc, char const *argv[]) {
    int n = 65;
    int nn = n*n;
    float* FDmat;
    float *bc, *u, *uini;
    float residual;
    bc   = (float*) calloc(nn,sizeof(float));
    u    = (float*) calloc(nn,sizeof(float));
    uini = (float*) calloc(nn,sizeof(float));
    float *A, *b, *u0;
    A  = (float*) malloc(nn*nn*sizeof(float));
    b  = (float*) malloc(nn*sizeof(float));
    u0 = (float*) malloc(nn*sizeof(float));
    if (!bc || !u || !uini || !A || !b || !u0) {
        perror("insufficient memory");
        printf("memory allocation error at line %d\n",__LINE__);
        return 1;
    }
    // iteration matrix genration
    if ( nineptr_mat(&FDmat, n, 1.0) != 0) {
        printf("memory allocation error at line %d\n",__LINE__);
        return 1;
    }
    // apply boundary value
    NorthBCAply(bc, n, exactsol, 'N', 1.0);

    // store the exact solution
    #ifdef DEBUGSOL
        EXACT = (float*) malloc(nn*sizeof(float));
        if (!EXACT) {
            perror("insufficient memory");
            printf("memory allocation error at line %d\n",__LINE__);
            return 1;
        }
        printf("=========== EXACT SOLUTION =============\n");
        float h = 1.0/(n+1);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                //     col
                // row -+---->x
                //      |
                //      y
                EXACT[ID(i,j,n)] = exactsol((j+1.0)*h, (i+1.0)*h);
                printf("%e\t", EXACT[ID(i,j,n)]);
            }
            printf("\n");
        }
        printf("=======================================\n");
    #endif
    // solve the Ax=b
    //--> Jocobi
    memcpy(A , FDmat, nn*nn*sizeof(float));
    memcpy(b , bc, nn*sizeof(float));
    memcpy(u0, uini, nn*sizeof(float));
    memset(u, 0, nn*sizeof(float));
    printf("======= Jocobi ======\n");
    //residual = Iterate_Jocobi(A, b, &u, &u0, nn);
    printf("=====================\n");
    //--> GS
    memcpy(A , FDmat, nn*nn*sizeof(float));
    memcpy(b , bc, nn*sizeof(float));
    memcpy(u0, uini, nn*sizeof(float));
    memset(u, 0, nn*sizeof(float));
    printf("======= Gauss Seidel ======\n");
    // residual = Iterate_GS(A, b, u, u0, nn);
    printf("===========================\n");
    //--> SOR
    memcpy(A , FDmat, nn*nn*sizeof(float));
    memcpy(b , bc, nn*sizeof(float));
    memcpy(u0, uini, nn*sizeof(float));
    memset(u, 0, nn*sizeof(float));
    printf("====== SOR ======\n");
    residual = Iterate_SOR(A, b, &u, &u0, nn, 1.85);
    printf("=================\n");

    #ifdef DEBUG
        for (size_t i = 0; i < nn; i++) {
            for (size_t j = 0; j < nn; j++) {
                printf("%6.1f\t", FDmat[ID(i,j,nn)]);
            }
            printf("\n");
        }
    #endif
    #ifdef DEBUGSOL
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                printf("%5.3e\t", u[ID(i,j,n)]);
            }
            printf("\n");
        }
    #endif
    free(A);
    free(b);
    free(u0);
    free(FDmat);
    free(bc);
    free(u);
    free(uini);
    return 0;
}
float absolute_err(float *u0, float *u1, size_t nn){
    float cvalue, infnorm=0.0;
    for (size_t i = 0; i < nn; i++) {
        cvalue = fabs(u0[i] - u1[i]);
        if (cvalue > infnorm) infnorm = cvalue;
    }
    return infnorm;
}
