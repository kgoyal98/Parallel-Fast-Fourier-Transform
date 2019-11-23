

g++ radix-2-serial.cpp -o radix-2-serial
nvcc radix-2-parallel.cu -o radix-2-parallel
g++ cooley-tukey-serial.cpp -o cooley-tukey-serial
nvcc cooley-tukey-parallel.cu -o cooley-tukey-parallel
nvcc cuFFT.cu -lcufft -o cufft
mkdir test
echo $'n\tradix-2-serial\tradix-2-parallel\tcuFFT'
for i in {4..10}
do
	n=$((2**$i))
	echo -n $n
	echo -n $'\t'
	python3 script.py -n "$n" -f inp
	./radix-2-serial < test/inp
	echo -n $'\t'
	./radix-2-parallel < test/inp
	echo -n $'\t'
	./cufft < test/inp
	echo -n $'\n'
done

echo $'n\tcooley-tukey-serial\tcooley-tukey-parallel\tcuFFT'
for i in {10..25}
do
	n=$((2**$i))
	echo -n $n
	echo -n $'\t'
	python3 script.py -n "$n" -f inp
	./cooley-tukey-serial < test/inp
	echo -n $'\t' 
	./cooley-tukey-parallel < test/inp
	echo -n $'\t'
	./cufft < test/inp
	echo -n $'\n'
done
