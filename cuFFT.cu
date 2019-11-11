#include <iostream>
#include <complex>
#include <math.h>
#include <thrust/complex.h>
#include <sys/time.h>
#include <cassert>
#include <cufft.h>

using namespace std;

int main(){
	int n;cin>>n;
	
	cufftComplex *data_host = (cufftComplex*) malloc (sizeof (cufftComplex)* n);
	cufftComplex *data_back = (cufftComplex*) malloc (sizeof (cufftComplex)* n);
	for(int i=0; i<n; i++){
		cin>>data_host[i].x;
		cin>>data_host[i].y;
	}
	cufftHandle plan;
	cufftComplex *data1;
	cudaMalloc ((void **) &data1, sizeof(cufftComplex)*n);
	cudaMemcpy(data1, data_host, n*sizeof(cufftComplex), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int batch=1;
	cufftPlan1d(&plan, n, CUFFT_C2C, batch);
	cufftExecC2C(plan, data1, data1, CUFFT_FORWARD);

	cudaEventRecord(stop);

	cudaMemcpy(data_back, data1, n*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<milliseconds;
	
	cufftDestroy(plan);
	// for(int i=0; i<n; i++){
		// cout<<"("<<data_back[i].x<<","<<data_back[i].y<<")"<<endl;					
	// }
}
