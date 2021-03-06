//http://stackoverflow.com/questions/22217628/integral-image-or-summed-area-table-of-2d-matrix-using-cuda-c
#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h> 
#include <stdio.h>
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
using namespace std;

__global__ void sat(int *a, int*b, int* flag,int rowsTotal,int colsTotal)
{
    // Thread Ids equal to block Ids because the each blocks contains one thread only.
    int col=blockDim.x*blockIdx.x+threadIdx.x;
    int row=blockDim.y*blockIdx.y+threadIdx.y;
        
    if (row>=rowsTotal || row <0) return;
    if (col>=colsTotal || col <0) return;

    int idx = threadIdx.y*blockDim.x+threadIdx.x; // id in block 
    int didx = row*colsTotal+col;  // compute data id
    __shared__ int s[BLOCK_DIM_X*BLOCK_DIM_Y];
    
    s[idx]=0;
    __syncthreads();

    //printf("run kernel\n");      
    
    while (s[BLOCK_DIM_X*BLOCK_DIM_Y-1]==0){
    if (s[idx]==0){
    if (threadIdx.x>0 && threadIdx.y==0)
    {
        if (s[idx-1])
        {
           if (row>0 )
              b[didx]=b[didx-colsTotal]+a[didx]+b[didx-1]-b[didx-colsTotal-1];
           else 
              b[didx]=a[didx]+b[didx-1];
           s[idx]=1;
        }
            
    }

    if (threadIdx.y>0 && threadIdx.x==0)
    {
        if (s[idx-blockDim.x])
        {
           if (col>0 )
              b[didx]=b[didx-colsTotal]+a[didx]+b[didx-1]-b[didx-colsTotal-1];
           else
              b[didx]=a[didx]+b[didx-colsTotal];
           s[idx]=1;
        }

    }
   
    if (threadIdx.y>0 && threadIdx.x>0)
    {
        if (s[idx-blockDim.x] && s[idx-1])
        {
           b[didx]=b[didx-colsTotal]+a[didx]+b[didx-1]-b[didx-colsTotal-1];
           s[idx]=1;
        }

    }
     if (threadIdx.x==0 && threadIdx.y==0)
     {  
       bool ready=false;
       if (blockIdx.x==0&&blockIdx.y==0)
            ready=true;
       if (blockIdx.x==0&&blockIdx.y>0)
       {
          if(flag[(blockIdx.y-1)*colsTotal/blockDim.x])
            ready=true;
       }
       if (blockIdx.x>0&&blockIdx.y==0)
       {
          if(flag[blockIdx.x-1])
            ready=true;
       }
       if (blockIdx.x>0&&blockIdx.y>0)
       {
          if(flag[blockIdx.y*colsTotal/blockDim.x+blockIdx.x-1] && flag[(blockIdx.y-1)*colsTotal/blockDim.x+blockIdx.x])
            ready=true;
       }
       if (ready)
       { 
          if (row>0 && col>0)
              b[didx]=b[didx-colsTotal]+a[didx]+b[didx-1]-b[didx-colsTotal-1];
          if (row==0 && col>0)
              b[didx]=a[didx]+b[didx-1];    
          if (row>0 && col==0)
             b[didx]=b[didx-colsTotal]+a[didx];
          if (row==0 && col==0)
              b[didx]=a[didx];
          s[idx]=1; 
       }
     }

    }
        __syncthreads();
    } 
    atomicExch(flag+blockIdx.y*colsTotal/blockDim.x+blockIdx.x,1);
    // for older version then cuda 3.0, comment out the above line and chnage it to the following.
    // flag[blockIdx.y*colsTotal/blockDim.x+blockIdx.x]=1;
    
}

void cpu_sat(int* a, int* b, int M, int N){

    for(int r=0;r<M;r++)
    {
        for(int c=0; c<N;c++)
        {
            if(r==0) 
            {
              if (c>0)
                  b[r*N+c]=b[r*N+c-1]+a[r*N+c];
              else
                  b[r*N+c]=a[r*N+c];
            }
            else{
              if (c>0)
                  b[r*N+c]=b[r*N+c-1]+a[r*N+c]+b[(r-1)*N+c]-b[(r-1)*N+c-1];
              else
                  b[r*N+c]=a[r*N+c]+b[(r-1)*N+c];
            }
        }
    }

}
int main()
{
    //M is number of rows
    //N is number of columns
    //M,N have to be multiples of BLOCK_DIM_X and BLOCK_DIM_Y
    int M=64,N=64;
    int total_e=M*N;
    int widthstep=total_e*sizeof(int);

    int * matrix_a= (int *)malloc(widthstep);
    int * matrix_b= (int *)malloc(widthstep);
    int * cpu_result = (int *)malloc(widthstep);
    int * h_flag = (int *)malloc(M*N/BLOCK_DIM_X/BLOCK_DIM_Y*sizeof(int));
    //cout<<"Enter elements for "<< M<<"x"<<N<<" matrix";

    for(int r=0;r<M;r++)
    {
        for(int c=0; c<N;c++)
        {
            //cout<<"Enter Matrix element [ "<<r<<","<<c<<"]";
            matrix_a[r*N+c]=rand()%100;
            matrix_b[r*N+c]=0;
        }
 
    }

    for (int i=0;i<M*N/BLOCK_DIM_X/BLOCK_DIM_Y;i++)
    {
        h_flag[i]=0;
    }
    cpu_sat(matrix_a,cpu_result,M,N);
    int * d_matrix_a, * d_matrix_b, * d_flag;

    //cout<<"start copy"<<endl;
    /*
    for(int r=0;r<M;r++)
    {
        for(int c=0; c<N;c++)
        {
            cout << matrix_a[r*N+c]<<" ";
        }
        cout << endl;
    }

    cout<<endl;
    */
    cudaMalloc(&d_matrix_a,widthstep);
    cudaMalloc(&d_matrix_b,widthstep);
    cudaMalloc(&d_flag,M*N/BLOCK_DIM_X/BLOCK_DIM_Y*sizeof(int));
    cudaMemcpy(d_matrix_a,matrix_a,widthstep,cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b,matrix_b,widthstep,cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag,h_flag,M*N/BLOCK_DIM_X/BLOCK_DIM_Y*sizeof(int),cudaMemcpyHostToDevice);

    //Creating a grid where the number of blocks are equal to the number of pixels or input matrix elements.

    //Each block contains only one thread.

    dim3 grid(N/BLOCK_DIM_X,M/BLOCK_DIM_Y);  // grid is two dimensional!!
    dim3 blockdim(BLOCK_DIM_X,BLOCK_DIM_Y);
    sat<<<grid,blockdim>>>(d_matrix_a, d_matrix_b,d_flag,M,N);
    //for (int i=0;i<M/BLOCK_DIM_Y+N/BLOCK_DIM_X-1;i++){
    //    sat<<<grid,blockdim>>>(d_matrix_a, d_matrix_b,M,N,i);
        //cudaThreadSynchronize();
    //}

    //cudaThreadSynchronize();
    cudaMemcpy(matrix_b,d_matrix_b,widthstep,cudaMemcpyDeviceToHost);
    cout<<"Compare with CPU result: "<<endl;
    int count=0; 
    for(int r=0;r<M;r++)
    {
        for(int c=0; c<N;c++)
        {
            if(cpu_result[r*N+c]!=matrix_b[r*N+c])
            {   
                count+=1;//cout << matrix_b[r*N+c]<<" "<<;
            //    if(r==0)
                cout<<r<<" "<<c<<" cpu:"<<cpu_result[r*N+c]<<" gpu:"<<matrix_b[r*N+c]<<endl;
            }
        }
        //cout << endl;
    }
    cout<<"mismatch: "<<count<<endl;
    //system("pause");

    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    free(matrix_a);
    free(matrix_b);
    return 0;
}
