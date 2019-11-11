#include <iostream>
#include <complex>
#include <math.h>
#include <ctime>

using namespace std;
complex<double> x[1<<20], y[1<<20];
complex<double> temp[1<<21];
int pout[1<<20], pin[1<<20];

complex<double> root_unity(int n, int k){
	return complex<double>(cos(-M_PI*2*k/n), sin(-M_PI*2*k/n));
}

void fft(int n){
	for(int i=0; i<n; i++){
		pout[i]=0;
		pin[i]=1;
		temp[pout[i]*n + i] = x[i];
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
		y[i] = temp[pout[i]*n + i];
	}
}

int main(){
	int n;
	cin>>n;
	for(int i=0; i<n; i++){
		int t,u; cin>>t>>u;
		x[i] = complex<double>(t, u);
	}
	clock_t cpu_startTime, cpu_endTime;
	double cpu_ElapseTime=0;
	cpu_startTime = clock();
	
	fft(n);

	cpu_endTime = clock();
	cpu_ElapseTime = double(cpu_endTime - cpu_startTime);
  	cout<<cpu_ElapseTime*1000/CLOCKS_PER_SEC;
	// for(int i=0; i<n; i++){
	// 	cout<<y[i]<<endl;
	// }
}
