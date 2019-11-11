#include <iostream>
#include <complex>
#include <math.h>
#include <thrust/complex.h>
#include <sys/time.h>

using namespace std;

__global__
void fft(thrust::complex<double> *g_odata, thrust::complex<double> *g_idata, int n)
{
	extern __shared__ thrust::complex<double> temp[]; // allocated on invocation
	int thid = threadIdx.x;
	int pout = 0, pin = 1;
	temp[pout*n + thid] = g_idata[thid];
	__syncthreads();
	int thid1=0;
	int b = __log2f(n+1);

	for(int i=0; i<b;i++){
		if(thid & (1<<i))
			thid1 |= (1<<(b-1-i));
	}
	pout = 1 - pout;
	pin = 1 - pin;
	temp[pout*n + thid] = temp[pin*n + thid1];
	__syncthreads();

	for(int i=1; i<n; i*=2){
		pout = 1 - pout;
		pin = 1 - pin;
		thid1 = thid ^ i;
		thrust::complex<double> factor(cos(-M_PI*thid/i), sin(-M_PI*thid/i));
		if(thid1 > thid){
			temp[pout*n + thid] = temp[pin*n + thid] + factor * temp[pin*n + thid1];
		}
		else{
			temp[pout*n + thid] = temp[pin*n + thid1] + factor * temp[pin*n + thid];
		}
		__syncthreads();
	}
	g_odata[thid] = temp[pout*n + thid];
}

void checkError(){
	cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) 
	  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
	  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

int main(void)
{
	int N;
	cin>>N;
	thrust::complex<double> *x, *y;
	cudaMallocManaged(&x, N*sizeof(thrust::complex<double>));
	cudaMallocManaged(&y, N*sizeof(thrust::complex<double>));
	
	for(int i=0; i<N;i++){
		int t,u; cin>>t>>u;
		x[i] = complex<double>(t, u);
	}
	int blockSize = N;
	int numBlocks = (N + blockSize - 1) / blockSize;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	fft<<<numBlocks, blockSize, 2*N*sizeof(thrust::complex<double>)>>>(y, x, N);
	checkError();
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<milliseconds;
	// for (int i = 0; i < N; i++)
		// cout<<y[i]<<"\n";
	// cout<<endl;
	cudaFree(x);
	cudaFree(y);
}
