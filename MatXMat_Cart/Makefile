all: main funciones compare generate

main: main.cpp Matrix.h fun.h
	@mpicxx -O3 main.cpp -o main

generate: funciones.cpp
	@g++ -O3 funciones.cpp -o generate

print: generate
	./generate A
	
compare: compare.cpp fun.h
	@g++ -O3 compare.cpp -o compare
	
testreport: main 
	./generate 512 512 A
	./generate 512 512 B
	sleep 5
	mpisubmit.bg -n 1 -w 00:05:00 -m SMP main -- A B C 1
	sleep 100
	mpisubmit.bg -n 8 -w 00:05:00 -m SMP main -- A B C1 2
	sleep 100
	./compare C C1
	mpisubmit.bg -n 64 -w 00:05:00 -m SMP main -- A B C2 4
	sleep 100
	./compare C C2
