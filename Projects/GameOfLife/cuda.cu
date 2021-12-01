#include <cuda_runtime.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>


void liveCell(int x, int y, bool arrayInput[], int x_size, int y_size){
	arrayInput[y*x_size+x] = true;
}

bool* parseRLE(char* inputRLE, int x_start, int y_start, bool arrayInput[], int x_size, int y_size, int &x_search, int &y_search){
	int i,x,y;
	int lastIndex = 0;
	int number = 0;
	int numberOfRows = 0;
	int rowNumber = 0;
	int rowLength = 0;
	int maxLength = 0;
	bool end = false;
	bool alloc = false;

	bool** arr;

    bool alive = false;
	for(i = 0; i < strlen(inputRLE)+1;i++){
		if(end){
			if(alloc)
				break;
			else{
				alloc = true;
				arr = (bool**)malloc(numberOfRows * sizeof(bool*));
				for(i = 0;i<numberOfRows;i++)
        			arr[i] = (bool*)malloc(maxLength * sizeof(bool));
				i = 0;
				for(x=0;x<numberOfRows;x++){
					for(y=0;y<maxLength;y++){
						arr[x][y] = false;
					}
				}
				lastIndex = 0;
				rowNumber = 0;
				end = false;
			}
		}

		alive = false;
		switch(inputRLE[i])
		{
			case 'o':
				alive = true;
			case 'b':
				if(lastIndex == i){
					number = 1;
				}
				else{
					number = strtol(&inputRLE[lastIndex], NULL, 10);
				}

				if(alloc && alive){
					for(x=rowLength;x<rowLength+number;x++){
						arr[rowNumber][x] = true;
					}
				}

				rowLength+=number;
				number = 0;
				lastIndex = i+1;
				break;
			case '!':
        		end = true;
			case '$':
				rowNumber++;
				if(!alloc)
					numberOfRows++;
        		if(maxLength<rowLength)
        			maxLength=rowLength;
				rowLength = 0;
				lastIndex++;
				break;
			default:
				break;
		}
	}

	bool *output = (bool*)malloc(maxLength * numberOfRows * sizeof(bool));

	for(x=0;x<maxLength;x++){
		for(y=0;y<numberOfRows;y++){
			output[y*maxLength+x] = arr[y][x];
			if(arr[y][x])
				liveCell(x_start+x,y_start+y,arrayInput,x_size,y_size);
		}
	}

	x_search = maxLength;
	y_search = numberOfRows;
	return output;
}

void preparePrintSearch(int searchOutputP[], int x_size, int y_size, int *searchSizes){
	for(int i = 0; i < x_size; i++){
		for(int j = 0; j < y_size; j++){
			if(searchOutputP[j*x_size+i]>=0){
				int x_searchSize = searchSizes[searchOutputP[j*x_size+i]*2];
				int y_searchSize = searchSizes[searchOutputP[j*x_size+i]*2+1];

				for(int x = 1;x<x_searchSize+1;x++){
					searchOutputP[(j)*x_size+i+x]=-2;
					searchOutputP[(j+1+y_searchSize)*x_size+i+x]=-2;
				}

				for(int y = 1;y<y_searchSize+1;y++){
					searchOutputP[(j+y)*x_size+i]=-3;
					searchOutputP[(j+y)*x_size+i+1+x_searchSize]=-4;
				}

				searchOutputP[(j)*x_size+i]=-5;
				searchOutputP[(j)*x_size+i+1+x_searchSize]=-6;
				searchOutputP[(j+1+y_searchSize)*x_size+i]=-5;
				searchOutputP[(j+1+y_searchSize)*x_size+i+1+x_searchSize]=-6;
			}
		}
	}
}

void printArray(bool arrayInputP[], int x_size, int y_size, int searchInputP[], bool search = false){
	printf("+");
	for(int x = 0; x < x_size; x++)
		printf("--");
	printf("+\n");

	for(int x = 0; x < x_size*y_size; x++){
		if(x%x_size==0)
			printf("|");

		if(search && searchInputP[x] == -2)
			printf("--");
		else if(search && searchInputP[x] == -3)
			printf(" |");
		else if(search && searchInputP[x] == -4)
			printf("| ");
		else if(search && searchInputP[x] == -5)
			printf(" +");
		else if(search && searchInputP[x] == -6)
			printf("+ ");
		else if(arrayInputP[x])
			printf("■ ");
		else
			printf(" ⠀");

		if(x%x_size==x_size-1)
			printf("|\n");
	}

	printf("+");
	for(int x = 0; x < x_size; x++)
		printf("--");
	printf("+\n");
}

__global__ void kernel_gof(bool arrayInputP[], bool arrayOutputP[], int x_size, int y_size){
	//printf("[%d;%d]\n",threadIdx.x,threadIdx.y);
	int kernelPos = blockDim.x * blockIdx.x + threadIdx.x;
	if(kernelPos>=y_size*x_size) return;
	int currY = kernelPos/x_size;
	int currX = kernelPos%x_size;

	bool live = false;
	int liveCount = 0;
	int startY = currY - 1;
	if(currY == 0) startY++;
	int startX = currX - 1;
	if(currX == 0) startX++;
	int endY = currY + 1;
	if(currY == y_size-1) endY--;
	int endX = currX + 1;
	if(currX == x_size-1) endX--;


	for(int y = startY; y <= endY; y++)
		for(int x = startX; x <= endX; x++)
				if(arrayInputP[y*x_size+x])
					liveCount++;

	if(arrayInputP[currY*x_size+currX]) liveCount--;

	if(liveCount == 3)
		live = true;
	if(arrayInputP[currY*x_size+currX] && liveCount == 2)
		live = true;
	//bool liveBefore = arrayInputP[threadIdx.y*y_size+threadIdx.x];
	//printf("[%d;%d]:[%d-%d;%d-%d]:%d,%d->%d\n",threadIdx.x,threadIdx.y,startX,endX,startY,endY,liveCount,liveBefore,live);

	arrayOutputP[currY*x_size+currX] = live;
}

__global__ void kernel_gofSearch(bool arrayInputP[], bool searchInputP[], int searchOutputP[], int searchSizesP[], int searchCount, int x_size, int y_size){
	int kernelPos = blockDim.x * blockIdx.x + threadIdx.x;
	if(kernelPos>=y_size*x_size) return;
	int currY = kernelPos/x_size;
	int currX = kernelPos%x_size;
	int output = -1; 
	int searchIndex = 0;

	for(int i = 0; i < searchCount; i++){
		bool *searchInputTemp = &searchInputP[searchIndex];
		int x_searchSize = searchSizesP[i*2];
		int y_searchSize = searchSizesP[i*2+1];
		if(currX+x_searchSize+2<x_size && currY+y_searchSize+2<y_size ){
			bool found = true;
			for(int y = 0;y<y_searchSize+2;y++){
				if(arrayInputP[(currY+y)*x_size+(currX)]){
					found = false;
					break;
				}
				if(arrayInputP[(currY+y)*x_size+(currX+x_searchSize+1)]){
					found = false;
					break;
				}
			}

			for(int x = 0;x<x_searchSize+2;x++){
				if(arrayInputP[(currY)*x_size+(currX+x)]){
					found = false;
					break;
				}
				if(arrayInputP[(currY+y_searchSize+1)*x_size+(currX+x)]){
					found = false;
					break;
				}
			}

			for(int y = 0;y<y_searchSize;y++){
				if(!found)
					break;
				for(int x = 0;x<x_searchSize;x++){
					if(arrayInputP[(currY+1+y)*x_size+(currX+1+x)] != searchInputTemp[y*x_searchSize+x]){
						found = false;
						break;
					}
				}
			}
			if(found){
				output = i;
				break;
			}
		}

		searchIndex += x_searchSize*y_searchSize;
	}

	searchOutputP[currY*x_size+currX] = output;
}

void ErrorCheck(cudaError_t cerr){
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	
}

int run_gof(int argc, char **argv){
	int x_size = 50;
	int y_size = 20;

	int cycles = 0;
	int cyclesAnimate = 200;
	if (argc == 2) {
		cycles = -atoi(argv[1]);
		cyclesAnimate = 1;
	}

	if (argc == 3) {
		x_size = atoi(argv[1]);
		y_size = atoi(argv[2]);
	}

	int searchCount = 3, x_searchSize, y_searchSize, searchInputSize=0;
	char search[searchCount][100]={
		"bo$2bo$3o!\0",
		"2bo$obo$b2o!\0",
		"2o$2o!\0"
	};

	int *searchSizesP;
	int *searchSizes = (int*) malloc(searchCount*2*sizeof(int));

	int arraySize = x_size*y_size;
	bool *arrayInputP;
	bool *arrayInput = (bool*) malloc(arraySize* sizeof(bool));

	bool *searchInputTemp;
	bool *searchInputP;
	bool *searchInput = (bool*) malloc(sizeof(bool));

	for(int i=0;i<searchCount;i++){
		searchInputTemp = parseRLE(search[i], 0, 0, arrayInput, x_size, y_size, x_searchSize, y_searchSize);
		searchInput = (bool*)realloc(searchInput, (searchInputSize+x_searchSize*y_searchSize)*sizeof(bool));

		for(int j=0;j<x_searchSize*y_searchSize;j++)
			searchInput[searchInputSize++]=searchInputTemp[j];

		searchSizes[i*2] = x_searchSize;
		searchSizes[i*2+1] = y_searchSize;
	}


	int searchLast=0;
	for(int i=0;i<searchCount;i++){
		printArray(&searchInput[searchLast], searchSizes[2*i], searchSizes[2*i+1], NULL);
		searchLast+=searchSizes[2*i]*searchSizes[2*i+1];
	}

	for(int y = 0; y < x_size*y_size; y++)
		arrayInput[y] = false;

	char input[] = "24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$10bo5bo7bo$11bo3bo$12b2o!\0";
	parseRLE(input, 5, 5, arrayInput, x_size, y_size, x_searchSize, y_searchSize);

	printArray(arrayInput, x_size, y_size, NULL);

	ErrorCheck(cudaMalloc(&arrayInputP, arraySize * sizeof(bool)));
	ErrorCheck(cudaMemcpy( arrayInputP, arrayInput, arraySize * sizeof(bool), cudaMemcpyHostToDevice ));

	ErrorCheck(cudaMalloc(&searchInputP, searchInputSize * sizeof(bool)));
	ErrorCheck(cudaMemcpy( searchInputP, searchInput, searchInputSize * sizeof(bool), cudaMemcpyHostToDevice ));

	ErrorCheck(cudaMalloc(&searchSizesP, arraySize * sizeof(int)));
	ErrorCheck(cudaMemcpy( searchSizesP, searchSizes, searchCount * 2 * sizeof(int), cudaMemcpyHostToDevice ));

	bool *arrayOutputP;
	bool *arrayOutput = (bool*) malloc(arraySize * sizeof(bool));
	ErrorCheck(cudaMalloc(&arrayOutputP, arraySize * sizeof(bool)));

	int *searchOutputP;
	int *searchOutput = (int*) malloc(arraySize * sizeof(int));
	ErrorCheck(cudaMalloc(&searchOutputP, arraySize * sizeof(int)));

	int blocks = ceil(x_size*y_size/100);
	int threads = x_size*y_size/(blocks);

	printf("Running with %d threads in %d blocks.\n", threads, blocks);

	for(; cycles < cyclesAnimate; cycles++){
		kernel_gof<<< blocks, threads >>>(arrayInputP, arrayOutputP, x_size, y_size);
		kernel_gofSearch<<< blocks, threads >>>(arrayOutputP, searchInputP, searchOutputP, searchSizesP, searchCount, x_size, y_size);

		if(cycles >= 0){
			ErrorCheck(cudaMemcpy( arrayOutput, arrayOutputP, arraySize * sizeof(bool), cudaMemcpyDeviceToHost ));
			ErrorCheck(cudaMemcpy( searchOutput, searchOutputP, arraySize * sizeof(int), cudaMemcpyDeviceToHost ));

			preparePrintSearch(searchOutput, x_size, y_size, searchSizes);
			printArray(arrayOutput, x_size, y_size, searchOutput, true);
			usleep(100*1000);
		}

		bool *temp = arrayInputP;
		arrayInputP = arrayOutputP;
		arrayOutputP = temp;
	}//while(getchar() != '\n');

	//cudaDeviceSynchronize();

	// Free memory
	cudaFree( arrayInputP );
	cudaFree( arrayOutputP );
	return 0;
}
