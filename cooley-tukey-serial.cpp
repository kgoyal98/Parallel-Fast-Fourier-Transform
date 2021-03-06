#include <iostream>
#include <complex>
#include <math.h>
#include <ctime>
#include <cassert>

using namespace std;
int block_1 = 16;
int block_2 = 8;


complex<double> x[1<<20], y[1<<20];
complex<double> temp[1<<21];
int pout[1<<20], pin[1<<20];

complex<double> root_unity(int n, int k){
	return complex<double>(cos(-M_PI*2*k/n), sin(-M_PI*2*k/n));
}

void fft(int blidx, int blidy, int l, int n){
	int k = blidx*l+blidy;
	int step = l/n;
	complex<double> temp[2*n];
	int pout[n], pin[n];
	for(int i=0; i<n; i++){
		pout[i]=0;
		pin[i]=1;
		temp[pout[i]*n + i] = x[k+i*step];
	}
	int thid1;
	for(int thid=0; thid<n; thid++){
		thid1 = 0;
		int b = log2(n+1);
		for(int i=0; i<b;i++){
		  if(thid & (1<<i))
		    thid1 |= (1<<(b-1-i));
		}
		pout[thid] = 1 - pout[thid];
		pin[thid] = 1 - pin[thid];
		temp[pout[thid]*n + thid] = temp[pin[thid]*n + thid1];
	}

	for(int i=1; i<n; i*=2){
		for(int thid=0; thid<n; thid++){
			pout[thid] = 1 - pout[thid];
			pin[thid] = 1 - pin[thid];
			thid1 = thid ^ i;
			complex<double> factor(cos(-M_PI*thid/i), sin(-M_PI*thid/i));
			if(thid1 > thid){
				temp[pout[thid]*n + thid] = temp[pin[thid]*n + thid] + factor * temp[pin[thid]*n + thid1];
			}
			else{
				temp[pout[thid]*n + thid] = temp[pin[thid]*n + thid1] + factor * temp[pin[thid]*n + thid];
			}
		}
	}
	for(int i=0; i<n; i++){
		complex<double> factor = root_unity(l, i*blidy);
		x[k+i*step] = factor * temp[pout[i]*n + i];
	}
}

void fft_last(int blidx, int x1, int x2){
	int l=block_1, n=block_1, blidy=0;
	int k = blidx*l+blidy;
	int step = l/n;
	for(int i=0; i<n; i++){
		pout[i]=0;
		pin[i]=1;
		temp[pout[i]*n + i] = x[k+i*step];
	}
	int thid1;
	for(int thid=0; thid<n; thid++){
		thid1 = 0;
		int b = log2(n+1);
		for(int i=0; i<b;i++){
		  if(thid & (1<<i))
		    thid1 |= (1<<(b-1-i));
		}
		pout[thid] = 1 - pout[thid];
		pin[thid] = 1 - pin[thid];
		temp[pout[thid]*n + thid] = temp[pin[thid]*n + thid1];
	}

	for(int i=1; i<n; i*=2){
		for(int thid=0; thid<n; thid++){
			pout[thid] = 1 - pout[thid];
			pin[thid] = 1 - pin[thid];
			thid1 = thid ^ i;
			complex<double> factor(cos(-M_PI*thid/i), sin(-M_PI*thid/i));
			if(thid1 > thid){
				temp[pout[thid]*n + thid] = temp[pin[thid]*n + thid] + factor * temp[pin[thid]*n + thid1];
			}
			else{
				temp[pout[thid]*n + thid] = temp[pin[thid]*n + thid1] + factor * temp[pin[thid]*n + thid];
			}
		}
	}

	for(int i=0; i<n; i++){
		int p = blidx;
		int j = i;
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
		y[loc] = temp[pout[i]*n + i];
	}
}

void fft(int n){
	int m = log2(n+1);
	int x1,x2, log_block_1=log2(block_1+1), log_block_2=log2(block_2+1);
	for(int i=0; i<log_block_1; i++){
		if((m-log_block_2*i)%log_block_1 == 0){
			x1 = (m-log_block_2*i)/log_block_1;
			x2=i;
		}
	}

	int l = n;
	for(int i=0; i<x1-1; i++){
		for(int blidx=0; blidx<n/l; blidx++){
			for(int blidy=0; blidy<l/block_1; blidy++){
				fft(blidx, blidy, l, block_1);
			}
		}	
		l/=block_1;
	}
	for(int i=0; i<x2; i++){
		for(int blidx=0; blidx<n/l; blidx++){
			for(int blidy=0; blidy<l/block_2; blidy++){
				fft(blidx, blidy, l, block_2);
			}
		}
		l/=block_2;
	}
	assert(l==block_1);
	for(int blidx=0; blidx<n/l; blidx++){
		fft_last(blidx, x1, x2);
	}
}


int main(){
	int n;
	cin>>n;
	for(int i=0; i<n; i++){
		int t,u; cin>>t>>u;
		x[i] = complex<double>(t, u);
		y[i] = complex<double>(t, u);
	}
	clock_t cpu_startTime, cpu_endTime;
	double cpu_ElapseTime=0;
	cpu_startTime = clock();

	fft(n);

	cpu_endTime = clock();
	cpu_ElapseTime = double(cpu_endTime - cpu_startTime);
  	cout<<cpu_ElapseTime*1000/CLOCKS_PER_SEC;
	// for(int i=0; i<n; i++){
		// cout<<y[i]<<"\n";
	// }
}