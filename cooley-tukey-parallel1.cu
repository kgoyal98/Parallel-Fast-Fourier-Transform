#include <iostream>
#include <complex>
#include <math.h>
#include <thrust/complex.h>
#include <sys/time.h>
#include <cassert>

using namespace std;

__constant__ const int block_1 = 16;
__constant__ const int block_2 = 8;


__global__
void fft_16(thrust::complex<double> *x, int l){
	const int n = block_1;
	int blidx = blockIdx.x, blidy = blockIdx.y;
	int k = blidx*l+blidy;
	int step = l/n;
	extern __shared__ thrust::complex<double> temp[];
	int thid = threadIdx.x;
	int pout=0, pin=1;
	temp[pout*n + thid] = x[k+thid*step];
	__syncthreads();
	int thid1;
	thid1 = 0;
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
	thrust::complex<double> factor = thrust::complex<double>(cos(-M_PI*2*thid*blidy/l), sin(-M_PI*2*thid*blidy/l));
	x[k+thid*step] = factor * temp[pout*n + thid];
}

__global__
void fft_8(thrust::complex<double> *x, int l){
	const int n = block_2;
	int blidx = blockIdx.x, blidy = blockIdx.y;
	int k = blidx*l+blidy;
	int step = l/n;
	extern __shared__ thrust::complex<double> temp[];
	int thid = threadIdx.x;
	int pout=0, pin=1;
	temp[pout*n + thid] = x[k+thid*step];
	__syncthreads();
	int thid1;
	thid1 = 0;
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
	thrust::complex<double> factor = thrust::complex<double>(cos(-M_PI*2*thid*blidy/l), sin(-M_PI*2*thid*blidy/l));
	x[k+thid*step] = factor * temp[pout*n + thid];
}

__global__
void fft_last(thrust::complex<double> *x, thrust::complex<double> *y, int x1, int x2){
	const int n = block_1;
	int blidx = blockIdx.x;
	int l=block_1, blidy=0;
	int k = blidx*l+blidy;
	int step = l/n;
	extern __shared__ thrust::complex<double> temp[];
	int thid = threadIdx.x;
	int pout=0, pin=1;
	temp[pout*n + thid] = x[k+thid*step];
	__syncthreads();
	int thid1;
	thid1 = 0;
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

	int p = blidx;
	int j = thid;
	int loc = j;
	for(int k=0; k<x2; k++){
		int t = p&(block_2-1);
		loc = loc*block_2+t;
		p = p/block_2;
	}
	for(int k=0; k<x1-1; k++){
		int t = p&(block_1-1);
		loc = loc*block_1+t;
		p = p/block_1;
	}

	y[loc] = temp[pout*n + thid];
}

void checkError(){
	cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) 
	  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
	  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}


int main(){
	int n;
	cin>>n;
	thrust::complex<double> *x, *y;
	cudaMallocManaged(&x, n*sizeof(thrust::complex<double>));
	cudaMallocManaged(&y, n*sizeof(thrust::complex<double>));
	for(int i=0; i<n; i++){
		int t,u; cin>>t>>u;
		x[i] = thrust::complex<double>(t, u);
	}
	int m = log2(n+1);
	clock_t cpu_startTime, cpu_endTime;
	double cpu_ElapseTime=0;
	cpu_startTime = clock();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	int x1,x2, log_block_1=log2(block_1+1), log_block_2=log2(block_2+1);
	for(int i=0; i<log_block_1; i++){
		if((m-log_block_2*i)%log_block_1 == 0){
			x1 = (m-log_block_2*i)/log_block_1;
			x2=i;
		}
	}
	// cout<<x1<<" "<<x2<<endl;
	int l = n;
	for(int i=0; i<x1-1; i++){
		dim3 grid(n/l, l/block_1, 1);
		dim3 block(16,1,1);
		fft_16<<<grid, block, 2*block_1*sizeof(thrust::complex<double>)>>>(x, l);
		// checkError();
		l/=block_1;
	}
	// print(x, n);
	for(int i=0; i<x2; i++){
		dim3 grid(n/l, l/block_2, 1);
		dim3 block(8,1,1);
		fft_8<<<grid, block, 2*block_2*sizeof(thrust::complex<double>)>>>(x, l);
		// checkError();
		l/=block_2;
	}
	assert(l==block_1);
	// print(x, n);
	// cout<<"l "<<l<<endl;
	dim3 grid(n/l, 1, 1);
	dim3 block(16,1,1);
	fft_last<<<grid, block, 2*block_1*sizeof(thrust::complex<double>)>>>(x, y, x1, x2);
	// checkError();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	// cout<<milliseconds;
	// cpu_endTime = clock();
	// cpu_ElapseTime = (cpu_endTime - cpu_startTime);
	// cout<<cpu_ElapseTime;
	for(int i=0; i<n; i++){
		// if(i%(n/block_1)==0) cout<<endl;
		cout<<y[i]<<"\n";
	}
}