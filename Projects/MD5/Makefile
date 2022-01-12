PATH:=${PATH}:/usr/local/cuda/bin
CC:=nvcc
OPT:=-Xcompiler -fPIC

main: cuda.o main.o
	${CC} $^ -o $@
	rm *.o

cuda.o: cuda.cu
	${CC} ${OPT}  -c $^  -o $@

main.o: main.cpp
	${CC} ${OPT}  -c $^  -o $@

clean:
	rm -f main
