

__global__
void fft_16(thrust::complex<double> *x, int l){
	int blidx = blockIdx.x, blidy = blockIdx.y;
	int k = blidx*l+blidy;
	int step = l/block_2;
	thrust::complex<double> temp[2*block_2];
	int pout[block_2], pin[block_2];
	// cout<<"fft-"<<n<<"\n";
	for(int i=0; i<block_2; i++){
		pout[i]=0;
		pin[i]=1;
		temp[pout[i]*block_2 + i] = x[k+i*step];
		// cout<<k+i*step<<" ";
	}
	// cout<<endl;
	int thid1;
	for(int thid=0; thid<block_2; thid++){
		thid1 = 0;
		int b = __log2f(block_2+1);
		for(int i=0; i<b;i++){
		  if(thid & (1<<i))
		    thid1 |= (1<<(b-1-i));
		}
		pout[thid] = 1 - pout[thid];
		pin[thid] = 1 - pin[thid];
		temp[pout[thid]*block_2 + thid] = temp[pin[thid]*block_2 + thid1];
	}

	for(int i=1; i<block_2; i*=2){
		for(int thid=0; thid<n; thid++){
			pout[thid] = 1 - pout[thid];
			pin[thid] = 1 - pin[thid];
			thid1 = thid ^ i;
			thrust::complex<double> factor(cos(-M_PI*thid/i), sin(-M_PI*thid/i));
			if(thid1 > thid){
				temp[pout[thid]*block_2 + thid] = temp[pin[thid]*block_2 + thid] + factor * temp[pin[thid]*block_2 + thid1];
			}
			else{
				temp[pout[thid]*block_2 + thid] = temp[pin[thid]*block_2 + thid1] + factor * temp[pin[thid]*block_2 + thid];
			}
		}
	}
	for(int i=0; i<block_2; i++){
		thrust::complex<double> factor = thrust::complex<double>(cos(-M_PI*2*i*blidy/l), sin(-M_PI*2*i*blidy/l));
		// cout<<temp[pout[i]*n + i]<<"* "<<"("<<i<<","<<blidy<<")\n";
		x[k+i*step] = factor * temp[pout[i]*n + i];
	}
	// cout<<"\n--------------------\n";
}
