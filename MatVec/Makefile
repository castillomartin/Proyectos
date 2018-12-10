all: main funciones compare generate

main: main.cpp Matrix.h fun.h
	@mpicxx -O3 main.cpp -o main

generate: funciones.cpp
	@g++ -O3 funciones.cpp -o generate

print: generate
	./generate A

compare: compare.cpp fun.h
	@g++ -O3 compare.cpp -o compare

	
	
report: main 
	./generate 512 512 A
	./generate 512 1 B
	sleep 5
	mpisubmit.bg -n 1 -w 00:05:00 -m SMP main -- A B C
	sleep 100
	mpisubmit.bg -n 32 -w 00:05:00 -m SMP main -- A B C1
	sleep 100
	./compare C C1
	sleep 5
	mpisubmit.bg -n 64 -w 00:05:00 -m SMP main -- A B C2
	sleep 100
	#./compare C C2
	#sleep 5
	mpisubmit.bg -n 128 -w 00:05:00 -m SMP main -- A B C3
	sleep 100
	#./compare C C3
	#sleep 5
	mpisubmit.bg -n 256 -w 00:05:00 -m SMP main -- A B C4
	sleep 100
	#./compare C C4
	#sleep 5
	mpisubmit.bg -n 512 -w 00:05:00 -m SMP main -- A B C5
	sleep 100
	#./compare C C5
	#sleep 5
	llsubmit testmap.jcf
	sleep 100
	