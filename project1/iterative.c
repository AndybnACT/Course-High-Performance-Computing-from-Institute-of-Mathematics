#include "iterative.h"

float absolute_err(float *u0, float *u1, size_t nn){
    float cvalue, infnorm=0.0;
    for (size_t i = 0; i < nn; i++) {
        cvalue = fabs(u0[i] - u1[i]);
        if (cvalue > infnorm) infnorm = cvalue;
    }
    return infnorm;
}

float Iterate_Jocobi(float *A, float *b, float **u, float **u_0, size_t nn){
    // List of variables will be written after this function:
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
