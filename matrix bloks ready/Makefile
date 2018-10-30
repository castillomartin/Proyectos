all: flops l1cache l2cache ipc tlb

flops: main.cpp Matriz.h
	g++ -I/usr/Inlcude -o0 main.cpp /usr/lib64/libpapi.so -o flops
	
l1cache: l1.cpp Matriz.h
	g++ -I/usr/Inlcude -o0 l1.cpp /usr/lib64/libpapi.so -o l1cache
	
l2cache: l2.cpp Matriz.h
	g++ -I/usr/Inlcude -o0 l2.cpp /usr/lib64/libpapi.so -o l2cache
	
ipc: cicle.cpp Matriz.h
	g++ -I/usr/Inlcude -o0 cicle.cpp /usr/lib64/libpapi.so -o ipc

tlb: tlb.cpp Matriz.h
	g++ -I/usr/Inlcude -o0 tlb.cpp /usr/lib64/libpapi.so -o tlb
	
testflops: cleanflops cleantime flops 
	./flops A B C 1
	./flops A B C 7 32
	./flops A B C 8 32
	./flops A B C 8 52
	
testl1cache: cleanl1 l1cache 
	./l1cache A B C 1
	./l1cache A B C 7 32
	./l1cache A B C 8 32
	./l1cache A B C 8 52
	
testl2cache: cleanl2 l2cache 
	./l2cache A B C 1
	./l2cache A B C 7 32
	./l2cache A B C 8 32
	./l2cache A B C 8 52
	
testipc: cleanipc ipc 
	./ipc A B C 1
	./ipc A B C 7 32
	./ipc A B C 8 32
	./ipc A B C 8 52

testtlb: cleantlb tlb 
	./tlb A B C 1
	./tlb A B C 7 32
	./tlb A B C 8 32
	./tlb A B C 8 52
	
compare: compare.cpp Matriz.h 
	g++ -o compare compare.cpp 
	
testcompare: cleanplot flops compare 
	./flops A B C 1
	./compare C C0
	./flops A B C 7 32
	./compare C C0
	./flops A B C 8 32
	./compare C C0
	
reportl1: l1.sh
	./l1.sh

reportl2: l2.sh
	./l2.sh

reportipc: ipc.sh
	./ipc.sh
	
reportflops: flops.sh
	./flops.sh

reporttime: time.sh
	./time.sh

clean:
	rm -f % *.bin *.o *.exe flops compare genera ipc tlb l1cache l2cache 
	
cleanplot:
	@rm -f % *.bin
	@touch plot.bin
	
cleanflops:
	@rm -f % flops.bin
	@touch flops.bin
	
cleanipc:
	@rm -f % ipc.bin
	@touch ipc.bin
	
cleanl1:
	@rm -f % l1.bin
	@touch l1.bin
	
cleanl2:
	@rm -f % l2.bin
	@touch l2.bin
	
cleantime:
	@rm -f % time.bin
	@touch time.bin