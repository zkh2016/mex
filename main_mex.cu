#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
//#include <mex.h>


/*****************************************************************************************************************
*------------------------------------------------headers.h--------------------------------------------------------*
/****************************************************************************************************************/

////////////////////////////////////////////////////////////////////////////////////
////GPU options
////////////////////////////////////////////////////////////////////////////////////

#define GRIDX_WIDTH		(MIN(256, NSYS))		
#define GPU_USE			0				//GPU device to use		//On local system, 0=gtx276, 1=gtx8800


////////////////////////////////////////////////////////////////////////////////////
////Linear systems options
////////////////////////////////////////////////////////////////////////////////////

#define NSYS	1			//Number of linear equation systems

#define N		256 //Rows (number of equations)
#define M		256 //Cols (number of variables)

////////////////////////////////////////////////////////////////////////////////////
////NNLS options
////////////////////////////////////////////////////////////////////////////////////

#define MAX_ITER_LS				((N+M)*2)
#define MAX_ITER_NNLS			((N+M)*2)

#define TOL_TERMINATION			(1e-10)
#define TOL_0					(1e-6)


////////////////////////////////////////////////////////////////////////////////////
////Debug options
////////////////////////////////////////////////////////////////////////////////////

#define DAT_DEBUG				1
#define DAT_DEBUG_CHECKROW		(0-NSTART)


////////////////////////////////////////////////////////////////////////////////////
////Macros
////////////////////////////////////////////////////////////////////////////////////

//Utility macros
#define MIN(a, b)	((a)>(b)?(b):(a))
#define MAX(a, b)	((a)>(b)?(a):(b))
#define SIGN(a)		((a)>0?1:-1)

//Index macros
#define X_I(a, b)		((a)*(N)+(b))
#define B_I(a, b)		((a)*(M)+(b))
#define R_I(a, b, c)	((a)*((M)*(M))+(b)*(M)+(c))
#define A_I(a, b, c)	((a)*((N)*(M))+(b)*(M)+(c))




/*****************************************************************************************************************
*------------------------------------------------utils.cu--------------------------------------------------------*
/****************************************************************************************************************/
__device__ float reduce512(float smem512[], unsigned short tID){
  __syncthreads();
  if(false){
      float sum = smem512[0];
      for(int i = 1; i < blockDim.x; i++){
          sum += smem512[i];
      }
      return sum;
  }else{
    int n = M;
    while(n/2){
        if(threadIdx.x < n / 2){
            float a = smem512[threadIdx.x];
            for(int i = threadIdx.x + n/2; i <  n; i+= n/2){
                a += smem512[i];
            }
            smem512[threadIdx.x] = a;
        }
        n/=2;
        __syncthreads();
    }
    return smem512[0];
  }
}


////////////////////////////////////////////////////////////
//Parallel maxIndex of 512 elements in 9 steps
////////////////////////////////////////////////////////////
__device__ int maxIndex512(float smem512[], float smemC512[], unsigned short tID){
  //smemC512[tID] = tID;
  __syncthreads();
  float max_v = smem512[0];
  float max_i = 0;
  for(int i = 1; i < blockDim.x; i++){
    if(max_v < smem512[i]){
        max_v = smem512[i];
        max_i = i;
    }
  }
  return max_i;
}



////////////////////////////////////////////////////////////
//Parallel l2 norm
////////////////////////////////////////////////////////////
__device__ float norml2_512(float smem512[], float *V, unsigned short tID){
  __syncthreads();
  smem512[tID] = V[tID] * V[tID];
  return sqrtf(reduce512(smem512, tID));

}

__device__ float norml2_512(float smem512[], float v, unsigned short tID){
  __syncthreads();
  smem512[tID] = v*v;
  return sqrtf(reduce512(smem512, tID));

}

__device__ float norml2sq512(float smem512[], float *V, unsigned short tID){
  __syncthreads();
  smem512[tID] = V[tID] * V[tID];
  return (reduce512(smem512, tID));

}

__device__ float norml2sq512(float smem512[], float v, unsigned short tID){
  __syncthreads();
  smem512[tID] = v*v;
  return (reduce512(smem512, tID));

}


////////////////////////////////////////////////////////////
//Parallel min of 512 elements in 9 steps
////////////////////////////////////////////////////////////
__device__ float min512(float smem512[], unsigned short tID){
  __syncthreads();
  float min = smem512[0];
  for(int i = 1; i < blockDim.x; i++){
    if(min > smem512[i]){
        min = smem512[i];
    }
  }
  return min;
}

/*****************************************************************************************************************
*------------------------------------------------nnls.cu--------------------------------------------------------*
/****************************************************************************************************************/

extern "C"{
  //////////////////////////////////////////////////////////////////////////////////////////
  //////NNLS (NxM) with QR modified Gram-Schmidt and Given's Rotation
  //////Lawson, C. L. and R. J. Hanson, Solving Least Squares Problems, Prentice-Hall, 1974, Chapter 23. 
  //////////////////////////////////////////////////////////////////////////////////////////

  __global__ void NNLS_MGS_GR_512(float *d_A, float *d_At, float *d_x, float *d_b,
    float *d_R, int *nIters, int *lsIters){

    __shared__ float	sDat[M];
    __shared__ float	sZ[M];
    __shared__ float	sB[M];
    __shared__ int		sK[M];

    __shared__ int nnlsCount;				//Number of NNLS iterations performed
    __shared__ int lsCount;					//Number of least-square subproblems solved

    unsigned short tID = threadIdx.x;		//thread index
    unsigned int gID = blockIdx.x * blockDim.x + threadIdx.x;		//thread index
    unsigned int sysID = blockIdx.y * GRIDX_WIDTH + blockIdx.x;
    __shared__ int sSysID;					//system index to solve
    __shared__ int kCols;					//k columns (elements) in set P

    float	x = 0;							//Solution initialized to 0
    bool	Z = true;						//Set Z initialized to 1 (in Z)

    float QtB;					//Q'b
    float Qt[M];				//Q'

    //Initialize variables with thread 0
    if (tID == 0){
      nnlsCount = 0;
      lsCount = 0;
      sSysID = sysID;
      kCols = 0;
    }
    __syncthreads();

    //Load B
    sB[tID] = d_b[B_I(sysID, tID)];

    //Run NNLS
    do{
      float Ax;								//Temp vars
      float Gx;
      ////Compute gradient of f=Ax-b and store into Gx
      for (short a = 0; a < N; ++a){
        __syncthreads();
        //Ax=sSig[min(max(midSig+a-tID, 0), SIG_BINS-1)]*XVal;
        Ax = d_A[A_I(sysID, a, tID)] * x;
        __syncthreads();
        sDat[tID] = Ax;
        Ax = reduce512(sDat, tID);
        if (tID == a) sZ[a] = sB[tID] - Ax;	//sZ temporary B-Ax
      }
      for (short a = 0; a < N; ++a){
        __syncthreads();
        //Ax = sSig[min(max(midSig+tID-a, 0), SIG_BINS-1)]*sZ[tID];
        Ax = d_At[A_I(sysID, a, tID)] * sZ[tID];
        __syncthreads();
        sDat[tID] = Ax;
        Ax = reduce512(sDat, tID);
        if (tID == a) Gx = Ax;					//Gx now holds A'(b-Ax) (gradient)
      }

      ////Check for termination condition
      __syncthreads();
      sDat[tID] = ((Z * (Gx > TOL_TERMINATION)) ? 1 : 0);
      bool wAllNonPositive = !((bool)reduce512(sDat, tID));
      if (wAllNonPositive)	break;					//Terminate

      //Find index of max(Gx) and remove from Z
      __syncthreads();
      sDat[tID] = (Z ? Gx : -FLT_MAX);
      unsigned short maxZinW = (unsigned short)maxIndex512(sDat, sZ, tID);
      if (tID == maxZinW)	Z = false;		//Update Z (remove index from Z and put into set P)
      __syncthreads();

      //Solve uncontrained linear least squares subproblem
      bool addColumn = true;

      do{
        if (tID == 0) ++lsCount;
        __syncthreads();

        if (addColumn){							//Adding a column via modified Gram-schmidt
          sZ[tID] = Z;						//Copy set Z (ZVal) into sZ

          //float newCol=sSig[min(max(midSig+tID-maxZinW, 0), SIG_BINS-1)];
          float newCol = d_At[A_I(sysID, maxZinW, tID)];
          __syncthreads();
          float oldCol = newCol;
          for (short k = 0; k < kCols; ++k){
            short a = sK[k];
            __syncthreads();
            Ax = Qt[a];
            __syncthreads();
            sDat[tID] = newCol*Ax;
            Gx = reduce512(sDat, tID);
            __syncthreads();
            newCol -= Ax*Gx;

            //Compute elements of R along column maxZinW
            sDat[tID] = oldCol*Ax;
            Gx = reduce512(sDat, tID);
            if (tID == 0) d_R[R_I(sysID, a, maxZinW)] = Gx;
          }

          __syncthreads();
          Gx = norml2_512(sDat, newCol, tID);	//Gx now ||V(a)||
          newCol /= Gx;
          __syncthreads();

          d_R[R_I(sysID, maxZinW, tID)] = (tID == maxZinW) * Gx;
          Qt[maxZinW] = newCol;

          //Compute QtB(a)=<Q(a), B>
          sDat[tID] = newCol*sB[tID];
          Gx = reduce512(sDat, tID);				//Gx now QtB(a)
          QtB = (tID == maxZinW) ? Gx : QtB;

          if (tID == 0)	sK[kCols++] = maxZinW;

          __syncthreads();

        }
        else{//Removing columns via Given's rotations

          //Partial QR factorization
          //Find which variables were removed from passive set
          bool removedVars = ((int)sZ[tID] != Z);
          __syncthreads();
          sZ[tID] = Z;				//Copy set Z (ZVal) into sZ

          __shared__ bool deletedCol;

          //Rebuild sK list
          for (short k = kCols - 1; k >= 0; --k){
            __syncthreads();
            short a = sK[k];

            if (tID == 0) deletedCol = false;
            __syncthreads();
            if (tID == a && removedVars) deletedCol = true;
            __syncthreads();

            if (deletedCol){
              //Perform given's rotation
              __shared__ float givenc;				//Given's Rotation coefficients c,f
              __shared__ float givens;

              Gx = d_R[R_I(sysID, a, tID)];			//Get row a (to zero out a/tc)

              for (short b = k + 1; b < kCols; ++b){
                __syncthreads();
                short tc = sK[b];					//Row & column (along diagonal) to zero out			

                __syncthreads();
                Ax = d_R[R_I(sysID, tc, tID)];		//Get row tc		
                //Gx = d_R[ R_I(sysID, a, tID) ];			//Get row a (to zero out a/tc)
                __syncthreads();

                //Compute c,s,-s,c coefficients if on row/column tc and row/column a/tc
                if (tID == tc){
                  //Computation avoids overflow
                  if (Ax == 0 && Gx == 0){
                    givenc = 0;
                    givens = 0;
                  }
                  else if (Ax == 0){
                    givenc = 0;
                    givens = 1;
                  }
                  else if (Gx == 0){
                    givenc = 1;
                    givens = 0;
                  }
                  else if (fabs(Gx) > fabs(Ax)){
                    float r = -Ax / Gx;
                    givens = 1 / sqrt(1 + r * r);
                    givenc = -givens * r;
                  }
                  else{
                    float r = -Gx / Ax;
                    givenc = 1 / sqrt(1 + r * r);
                    givens = -givenc * r;
                  }
                }
                __syncthreads();

                //Update R
                float tAx = Ax * givenc + Gx * givens;					//Update row tc
                float tGx = Ax * -givens + Gx * givenc;					//Update row a
                __syncthreads();
                d_R[R_I(sysID, tc, tID)] = tAx;						//Set row tc
                //d_R[ R_I(sysID, a, tID)] = tGx;							//Set row a
                Gx = tGx;

                //Update Q'
                __syncthreads();
                tAx = Qt[tc];											//Get row tc
                tGx = Qt[a];											//Get row a
                __syncthreads();
                Qt[tc] = tAx * givenc + tGx * givens;					//Update row tc
                Qt[a] = tAx * -givens + tGx * givenc;					//Update row a

                //Update Q'b
                __shared__ float bb, bt;
                if (tID == tc)		bt = QtB;							//Get row tc
                else if (tID == a)	bb = QtB;							//Get row a
                __syncthreads();
                if (tID == tc)		QtB = bt * givenc + bb * givens;	//Update row tc
                else if (tID == a)	QtB = bt * -givens + bb * givenc;	//Update row a
              }

              __syncthreads();
              d_R[R_I(sysID, a, tID)] = Gx;

              //Shift sK (column list in set P) list left by one
              __syncthreads();
              if (tID >= k && tID < kCols - 1) Ax = sK[tID + 1];
              __syncthreads();
              if (tID >= k && tID < kCols - 1) sK[tID] = Ax;
              __syncthreads();
              if (tID == 0)	kCols--;
            }
          }
          __syncthreads();
          //End of QR factorization
        }
        __syncthreads();

        //Compute solution z
        //Ax now z
        Ax = 0;
        for (short b = kCols - 1; b >= 0; --b){
          short a = sK[b];
          __syncthreads();

          //Compute z via backsubtitution
          float coeff = d_R[R_I(sysID, a, tID)] * !Z;

          __syncthreads();
          sDat[tID] = (!(tID == a)) * coeff * Ax;
          float pSum = reduce512(sDat, tID);

          if ((tID == a))
            Ax = ((coeff == 0) ? (0) : (QtB - pSum) / coeff);
        }
        __syncthreads();

        //Check subproblem
        //If smallest element in set P is positive, then accept
        sDat[tID] = (Z ? FLT_MAX : Ax);
        if (min512(sDat, tID) > 0){
          //Accept solution, update X
          x = Ax;
          break;
        }
        else{
          //Reject solution, update subproblem
          //Find index q in set P for negative z such that x/(x-z) is minimized
          __syncthreads();
          if (!Z && Ax <= 0)	sDat[tID] = x / (x - Ax);
          else				sDat[tID] = FLT_MAX;
          float alpha = min512(sDat, tID);

          //Update X
          x += alpha * (Ax - x);

          //Move from set P to set Z all elements whose corresponding X is 0 (guaranteed to have one element)
          Z = (Z) || (fabs(x) <= TOL_0);
          addColumn = false;
        }

      } while (lsCount < MAX_ITER_LS);

      if (lsCount >= MAX_ITER_LS)
        break;

      if (tID == 0) ++nnlsCount;
      __syncthreads();
    } while (nnlsCount < MAX_ITER_NNLS);

    //Output solution X and iteration counts
    __syncthreads();
    d_x[X_I(sSysID, tID)] = x;

    nIters[sSysID] = nnlsCount;
    lsIters[sSysID] = lsCount;

    //	nIters[sSysID] = stackColUpdated;
    //	lsIters[sSysID] = normalColUpdated;

  }
}

void from_file(float* a, float* b, const char* file_a, const char* file_b){
    FILE *fp_a = fopen(file_a, "r");
    FILE *fp_b = fopen(file_b, "r");
    if(fp_a == NULL|| fp_b == NULL){
        printf("open file failed!\n");
        return;
    }
    for(int k = 0; k < NSYS; k++){
        for(int i = 0; i < N; i++){
            for(int j = 0; j < M; j++){
                fscanf(fp_a, "%f,", &a[k * M * N + i * M + j]);
            }
        }
    }
    for(int i = 0; i < M * NSYS; i++){
        fscanf(fp_b, "%f", &b[i]);
    }
    fclose(fp_a);
    fclose(fp_b);
}


/*****************************************************************************************************************
*------------------------------------------------nnls.cu--------------------------------------------------------*
/****************************************************************************************************************/
int main(int argc, char** argv) {

  //Obtain GPU device info
  int deviceID = GPU_USE;
  cudaDeviceProp deviceProp;
  cudaSetDevice(deviceID);
  // char name[256];
  // cuDeviceGetName(name, 255, deviceID);
  // printf("Using device %s\n",name);

  cudaGetDeviceProperties(&deviceProp, deviceID);
  printf("regsPerBlock %d\n", deviceProp.regsPerBlock);
  printf("sharedMemPerBlock %zu\n", deviceProp.sharedMemPerBlock);
  printf("maxThreadsPerBlock %d\n", deviceProp.maxThreadsPerBlock);
  printf("maxThreadsDim %d %d %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
  printf("maxGridSize %d %d %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
  printf("total processor count %d\n", deviceProp.multiProcessorCount);
  printf("warpSize %d\n\n", deviceProp.warpSize);

  //Memory block sizes
  unsigned int a_matrix_mem_size = sizeof(float) * ((N * M) * NSYS);
  unsigned int x_mem_size = sizeof(float) * M * NSYS;
  unsigned int b_mem_size = sizeof(float) * N * NSYS;
  unsigned long int r_matrix_mem_size = sizeof(float) * M * M * NSYS;
  unsigned int iters_mem_size = sizeof(float) * NSYS;

  //Allocate gpu memory
  float *d_A, *d_At, *d_x, *d_b, *d_R;
  int *d_nnlsIters, *d_lsIters;
  cudaMalloc((void**)&d_A, a_matrix_mem_size);
  cudaMalloc((void**)&d_At, a_matrix_mem_size);
  cudaMalloc((void**)&d_x, x_mem_size);
  cudaMalloc((void**)&d_b, b_mem_size);
  cudaMalloc((void**)&d_R, r_matrix_mem_size);
  cudaMalloc((void**)&d_nnlsIters, iters_mem_size);
  cudaMalloc((void**)&d_lsIters, iters_mem_size);

  //Allocate host memory
  float *h_A = (float*)malloc(a_matrix_mem_size);
  float *h_At = (float*)malloc(a_matrix_mem_size);
  float *h_x = (float*)malloc(x_mem_size);
  float *h_b = (float*)malloc(b_mem_size);
  int *h_nnlsIters = (int*)malloc(iters_mem_size);
  int *h_lsIters = (int*)malloc(iters_mem_size);

  //Generate matrices A and vectors b
  if(false){
  srand(2010);
  for (int a = 0; a<NSYS; ++a){
    for (int b = 0; b<N; ++b){
      for (int c = 0; c<M; ++c){
        //Random
        //	h_A[ A_I(a,b,c) ] = (rand()%1000)/1000.0f;

        //Identity
        //h_A[ A_I(a,b,c) ] = (b == c);

        //Gaussian time-series
        float sigma = N / 100.0f;
        h_A[A_I(a, b, c)] = exp(-(b - c)*(b - c) / (2 * (sigma*sigma)));

        h_At[A_I(a, c, b)] = h_A[A_I(a, b, c)];
      }
    }
    //random vector b
    for (int b = 0; b<M; ++b)		h_b[B_I(a, b)] = (rand() % 1000) / 1000.0f;
  }
  }else{
    from_file(h_A, h_b, "System_Matrix_3D.csv", "3234.csv");
    for (int a = 0; a<NSYS; ++a){
        for (int b = 0; b<N; ++b){
            for (int c = 0; c<M; ++c){
                h_At[A_I(a, c, b)] = h_A[A_I(a, b, c)];
            }
        }
    }
  }

  //Timers
  // unsigned int timerMemcpyToGPU, timerGPU;
  // cutCreateTimer(&timerMemcpyToGPU);
  // cutCreateTimer(&timerGPU);
  float elapsed = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //GPU execution parameters
  dim3 dimBlock(M, 1, 1);
  dim3 dimGrid(GRIDX_WIDTH, (int)ceil((float)NSYS / GRIDX_WIDTH), 1);

  ////////////////////////////////////////////////////////////////////////////////
  //Begin Execution
  ////////////////////////////////////////////////////////////////////////////////

  //Copy host matrices A, vectors b, to GPU
  // cutStartTimer(timerMemcpyToGPU);
  cudaEventRecord(start, 0);
  cudaMemcpy(d_A, h_A, a_matrix_mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_At, h_At, a_matrix_mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, b_mem_size, cudaMemcpyHostToDevice);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  // cutStopTimer(timerMemcpyToGPU); 
  printf("Memcpy time: %f ms\n", elapsed);

  //Run GPU NNLS
  // cutStartTimer(timerGPU);
  cudaEventRecord(start, 0);
  NNLS_MGS_GR_512 << <dimGrid, dimBlock >> >(d_A, d_At, d_x, d_b,
    d_R, d_nnlsIters, d_lsIters);
  cudaError_t status = cudaDeviceSynchronize();
  if(status != cudaSuccess){
    printf("error: %s\n", cudaGetErrorString(status));
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  // cutStopTimer(timerGPU); 
  printf("NNLS execution time: %f ms\n", elapsed);

  //Get GPU results
  cudaMemcpy(h_x, d_x, x_mem_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_nnlsIters, d_nnlsIters, iters_mem_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_lsIters, d_lsIters, iters_mem_size, cudaMemcpyDeviceToHost);

  FILE *fp_x = fopen("h_x.csv", "w+");
  for(int i = 0; i < NSYS; i++){
    for(int j = 0; j < M; j++){
        fprintf(fp_x, "%f,", h_x[i * M + j]);
    }
  }
  fclose(fp_x);

  //Compute error ||Ax-b||2
  //Write x, Ax, and b
  int sum_nnlsIters = 0;
  int sum_lsIters = 0;
  FILE *f_gpu = fopen("gpu_result.txt", "w+");
  for (int a = 0; a < NSYS; ++a){
    float norm2 = 0;
    for (int b = 0; b < N; ++b){
      float temp = 0;
      for (int c = 0; c < M; ++c){
        temp += h_A[A_I(a, b, c)] * h_x[X_I(a, c)];
      }
      temp -= h_b[B_I(a, b)];
      norm2 += temp * temp;
    }
    norm2 = sqrt(norm2);
    printf("norm %f\n", norm2);

    sum_nnlsIters += h_nnlsIters[a];
    sum_lsIters += h_lsIters[a];
    fprintf(f_gpu, "\nSystem %d: nnlsIters %d, lsIters %d, norm2 %f\n", a, h_nnlsIters[a], h_lsIters[a], norm2);

    for (int b = 0; b < N; ++b) fprintf(f_gpu, "%f ", h_x[X_I(a, b)]);

    fprintf(f_gpu, "\n\nAx\n");
    for (int b = 0; b < N; ++b){
      float sum = 0;
      for (int c = 0; c < M; ++c)
        sum += h_A[A_I(a, b, c)] * h_x[X_I(a, c)];
      fprintf(f_gpu, "%f ", sum);
    }
    printf("\n");

    fprintf(f_gpu, "\n\n\n");
    for (int b = 0; b < N; ++b) fprintf(f_gpu, "%f ", h_b[B_I(a, b)]);

  }
  fprintf(f_gpu, "\ntotal nnlsIters %d, total lsIters %d\n", sum_nnlsIters, sum_lsIters);
  fclose(f_gpu);

  //Write A
  FILE *f_sysA = fopen("sysA.txt", "wb+");

  /*
  fprintf(f_sysA, "%d %d %d\n", NSYS, N, M);
  for(int a = 0; a < NSYS; ++a){
  for(int b = 0; b < N; ++b){
  for(int c = 0; c< M; ++c)
  fprintf(f_sysA, "%f ", h_A[ A_I(a, b, c ) ] );
  }
  }
  */
  fwrite(h_A, sizeof(float), NSYS*N*M, f_sysA);
  fclose(f_sysA);

  FILE *f_sysB = fopen("sysB.txt", "wb+");
  /*
  fprintf(f_sysB, "%d %d %d\n", NSYS, N, M);
  for(int a = 0; a < NSYS; ++a){
  for(int b = 0; b < N; ++b) fprintf(f_sysB, "%f ", h_b[ B_I(a, b) ]);
  }
  */
  fwrite(h_b, sizeof(float), NSYS*M, f_sysB);
  fclose(f_sysB);

  //Free host memory
  free(h_A);
  free(h_At);
  free(h_x);
  free(h_b);
  free(h_lsIters);
  free(h_nnlsIters);

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_At);
  cudaFree(d_x);
  cudaFree(d_b);
  cudaFree(d_R);
  cudaFree(d_nnlsIters);
  cudaFree(d_lsIters);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // CUT_EXIT(argc, argv);
}

/*
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  //get params from matlab
  double *h_A = mxGetPr(prhs[0]);
  double *h_b = mxGetPr(prhs[1]);

  //creat output pointer to matlab
  plhs[0] = mxCreateDoubleMatrix(1, M, mxREAL);
  double *h_x = mxGetPr(plhs[0]);

  //Obtain GPU device info
  int deviceID = GPU_USE;
  cudaDeviceProp deviceProp;
  cudaSetDevice(deviceID);
  cudaGetDeviceProperties(&deviceProp, deviceID);

  //Memory block sizes
  unsigned int a_matrix_mem_size = sizeof(float) * ((N * M) * NSYS);
  unsigned int x_mem_size = sizeof(float) * M * NSYS;
  unsigned int b_mem_size = sizeof(float) * N * NSYS;
  unsigned long int r_matrix_mem_size = sizeof(float) * M * M * NSYS;
  unsigned int iters_mem_size = sizeof(float) * NSYS;

  //Allocate gpu memory
  float *d_A, *d_At, *d_x, *d_b, *d_R;
  int *d_nnlsIters, *d_lsIters;
  cudaMalloc((void**)&d_A, a_matrix_mem_size);
  cudaMalloc((void**)&d_At, a_matrix_mem_size);
  cudaMalloc((void**)&d_x, x_mem_size);
  cudaMalloc((void**)&d_b, b_mem_size);
  cudaMalloc((void**)&d_R, r_matrix_mem_size);
  cudaMalloc((void**)&d_nnlsIters, iters_mem_size);
  cudaMalloc((void**)&d_lsIters, iters_mem_size);

  //Allocate host memory
  float *h_At = (float*)malloc(a_matrix_mem_size);
  int *h_nnlsIters = (int*)malloc(iters_mem_size);
  int *h_lsIters = (int*)malloc(iters_mem_size);

  //Generate matrices A and vectors b
  for (int a = 0; a<NSYS; ++a)
  {
    for (int b = 0; b<N; ++b)
    {
      for (int c = 0; c<M; ++c)
      {
        h_At[A_I(a, c, b)] = h_A[A_I(a, b, c)];
      }
    }
  }

  //Timers
  float elapsed = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //GPU execution parameters
  dim3 dimBlock(M, 1, 1);
  dim3 dimGrid(GRIDX_WIDTH, (int)ceil((float)NSYS / GRIDX_WIDTH), 1);

  //Copy host matrices A, vectors b, to GPU
  cudaEventRecord(start, 0);
  float *h_A2 = (float*)malloc(a_matrix_mem_size);
  float *h_b2 = (float*)malloc(b_mem_size);
  for(int i = 0; i < M*N*NSYS; i++){
    h_A2[i] = h_A[i];
  }
  for(int i = 0; i < N*NSYS;i++){
    h_b2[i] = h_b[i];
  }
  cudaMemcpy(d_A, h_A2, a_matrix_mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_At, h_At, a_matrix_mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b2, b_mem_size, cudaMemcpyHostToDevice);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  //Run GPU NNLS
  cudaEventRecord(start, 0);
  NNLS_MGS_GR_512 << <dimGrid, dimBlock >> >(
    d_A,
    d_At, 
    d_x,
    d_b,
    d_R,
    d_nnlsIters, 
    d_lsIters);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  //Get GPU results
  float *h_x2 = (float*)malloc(x_mem_size);
  cudaMemcpy(h_x2, d_x, x_mem_size, cudaMemcpyDeviceToHost);
  for(int i = 0; i < M*NSYS; i++){
    h_x[i] = h_x2[i];
  }
  cudaMemcpy(h_nnlsIters, d_nnlsIters, iters_mem_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_lsIters, d_lsIters, iters_mem_size, cudaMemcpyDeviceToHost);

  //Write x, Ax, and b
  int sum_nnlsIters = 0;
  int sum_lsIters = 0;
  FILE *f_gpu = fopen("gpu_result.txt", "w+");
  for (int a = 0; a < NSYS; ++a){
    float norm2 = 0;
    for (int b = 0; b < N; ++b){
      float temp = 0;
      for (int c = 0; c < M; ++c){
        temp += h_A[A_I(a, b, c)] * h_x[X_I(a, c)];
      }
      temp -= h_b[B_I(a, b)];
      norm2 += temp * temp;
    }
    norm2 = sqrt(norm2);
    printf("norm %f\n", norm2);

    sum_nnlsIters += h_nnlsIters[a];
    sum_lsIters += h_lsIters[a];
    fprintf(f_gpu, "\nSystem %d: nnlsIters %d, lsIters %d, norm2 %f\n", a, h_nnlsIters[a], h_lsIters[a], norm2);

    for (int b = 0; b < N; ++b) fprintf(f_gpu, "%f ", h_x[X_I(a, b)]);

    fprintf(f_gpu, "\n\nAx\n");
    for (int b = 0; b < N; ++b){
      float sum = 0;
      for (int c = 0; c < M; ++c)
        sum += h_A[A_I(a, b, c)] * h_x[X_I(a, c)];
      fprintf(f_gpu, "%f ", sum);
    }

    fprintf(f_gpu, "\n\nb\n");
    for (int b = 0; b < N; ++b) fprintf(f_gpu, "%f ", h_b[B_I(a, b)]);

  }
  fprintf(f_gpu, "\ntotal nnlsIters %d, total lsIters %d\n", sum_nnlsIters, sum_lsIters);
  fclose(f_gpu);

  //Write A
  FILE *f_sysA = fopen("sysA.txt", "wb+");
  fwrite(h_A, sizeof(float), NSYS*N*M, f_sysA);
  fclose(f_sysA);

  FILE *f_sysB = fopen("sysB.txt", "wb+");
  fwrite(h_b, sizeof(float), NSYS*M, f_sysB);
  fclose(f_sysB);

  //Free host memory
  free(h_A);
  free(h_At);
  free(h_x);
  free(h_b);
  free(h_lsIters);
  free(h_nnlsIters);
  free(h_A2);
  free(h_b2);
  free(h_x2);

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_At);
  cudaFree(d_x);
  cudaFree(d_b);
  cudaFree(d_R);
  cudaFree(d_nnlsIters);
  cudaFree(d_lsIters);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

}
*/
