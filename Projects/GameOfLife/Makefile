PATH:=${PATH}:/usr/local/cuda-10.2/bin
CC:=nvcc
OPT:=-Xcompiler -fPIC

main: cuda.o main.o
	${CC} $^ -o $@

cuda.o: cuda.cu
	${CC} ${OPT}  -c $^  -o $@

main.o: main.cpp
	${CC} ${OPT}  -c $^  -o $@

clean:
	rm *.o main
