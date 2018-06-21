#include "iterative.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float *EXACT;

int main(int argc, char const *argv[]) {
    size_t n = 10;
    float *A;
    float *b;
    float *u, *u0;
    float h = 1.0/n;

    A = (float*) calloc(n*n, sizeof(float));
    b = (float*) calloc(n, sizeof(float));
    u = (float*) calloc(n, sizeof(float));
    u0 = (float*) calloc(n, sizeof(float));
    EXACT = (float*) malloc(n*sizeof(float));
    for (size_t i = 0; i < n; i++) {
        EXACT[i] = ((float)i)/10.0;
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i == j) A[ID(i,j,n)] = -2.0;
            if (i%n != 0 && j == i-1 ) A[ID(i,j,n)] = 1.0;
            if (i%n != n-1 && j == i+1) A[ID(i,j,n)] = 1.0;
        }
    }
    b[n-1] = -1.0;

    Iterate_SOR(A,b,&u,&u0,n,1.1);

    for (size_t i = 0; i < n; i++) {
        printf("%e \n", u[i]);
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            printf("%6.1f\t", A[ID(i,j,n)]);
        }
        printf("\n");
    }
    return 0;
}
