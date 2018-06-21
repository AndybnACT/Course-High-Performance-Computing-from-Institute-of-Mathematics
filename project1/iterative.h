#include <stdio.h>
#include <math.h>

#define DEBUGSOL
#define JCBDEBUG
#define GSDEBUG
#define SORDEBUG
//#define float double
#ifdef DEBUGSOL
    extern float *EXACT;
#endif

#define PI 3.14159265359f
#define ID(row,col,n) row*n+col // row-major indecing
#define ITER_MAX 5000
#define ERRCRE 1e-4

float Iterate_Jocobi(float *, float *, float **, float **, size_t );
float Iterate_GS(float *, float *, float *, float *, size_t );
float Iterate_SOR(float *, float *, float **, float **, size_t , float );
