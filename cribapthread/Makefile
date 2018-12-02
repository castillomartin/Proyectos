main: main.cpp
	g++  main.cpp -o main -lm -pthread -lrt

test: clean
	./main 1 100 4

report: clean
	./main 1 10000000 1
	./main 1 10000000 2
	./main 1 10000000 4
	./main 1 10000000 8
	./main 1 10000000 16
	cat plot
	
clean:
	@rm -f % plot
	@touch plot
	
graph:
	./max.sh
	./total.sh