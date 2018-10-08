programa: main.o
	g++ -o programa main.o 

compare: compare.o 
	g++ -o compare compare.o
	
main.o: main.cpp Matriz.h
	g++ -c main.cpp

compare.o: compare.cpp Matriz.h
	g++ -c compare.cpp
	
main: programa
	./programa m500x500.bin m500x500.bin

test: compare
	./compare C1.bin C0.bin
	./compare C2.bin C0.bin
	./compare C3.bin C0.bin
	./compare C4.bin C0.bin
	./compare C5.bin C0.bin
	./compare C6.bin C0.bin

report: graficscript.sh
	./graficscript.sh
	
clean:
	rm -f % *.o *.exe