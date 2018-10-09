programa: main.o
	g++ -o programa main.o 

compare: compare.o 
	g++ -o compare compare.o
	
main.o: main.cpp Matriz.h
	g++ -c main.cpp

compare.o: compare.cpp Matriz.h
	g++ -c compare.cpp
	
test: cleanplot programa compare 
	./programa A B C 1
	./compare C C0
	./programa A B C 2
	./compare C C0
	./programa A B C 3
	./compare C C0
	./programa A B C 4
	./compare C C0
	./programa A B C 5
	./compare C C0
	./programa A B C 6
	./compare C C0

report: graficscript.sh
	./graficscript.sh
	
clean:
	rm -f % *.o *.exe
	
cleanplot:
	@rm -f % *.bin
	@touch plot.bin