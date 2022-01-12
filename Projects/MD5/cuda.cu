#include <cuda_runtime.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <chrono>
#include <iostream>

// Constants are the integer part of the sines of integers (in radians) * 2^32.
__device__ const uint32_t k[64] = {
	0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee ,
	0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501 ,
	0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be ,
	0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821 ,
	0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa ,
	0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8 ,
	0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed ,
	0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a ,
	0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c ,
	0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70 ,
	0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05 ,
	0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665 ,
	0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039 ,
	0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1 ,
	0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1 ,
	0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

// r specifies the per-round shift amounts
__device__ const uint32_t r[] = { 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
					  5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
					  4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
					  6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21 };

// leftrotate function definition
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))

__device__ void to_bytes2(uint32_t val, uint8_t* bytes) {
	bytes[0] = (uint8_t)val;
	bytes[1] = (uint8_t)(val >> 8);
	bytes[2] = (uint8_t)(val >> 16);
	bytes[3] = (uint8_t)(val >> 24);
}

__device__ uint32_t to_int32(const uint8_t* bytes) {
	return (uint32_t)bytes[0]
		| ((uint32_t)bytes[1] << 8)
		| ((uint32_t)bytes[2] << 16)
		| ((uint32_t)bytes[3] << 24);
}

__device__ void md5(const uint8_t* initial_msg, size_t initial_len, uint8_t* digest, uint8_t* msg) {

	// These vars will contain the hash
	uint32_t h0, h1, h2, h3;

	size_t offset;
	size_t new_len = 56;
	uint32_t w[16];
	uint32_t a, b, c, d, i, f, g, temp;

	// Initialize variables - simple count in nibbles:
	h0 = 0x67452301;
	h1 = 0xefcdab89;
	h2 = 0x98badcfe;
	h3 = 0x10325476;

	//Pre-processing:
	//append "1" bit to message    
	//append "0" bits until message length in bits ≡ 448 (mod 512)
	//append length mod (2^64) to message

	//for (new_len = initial_len + 1; new_len % (512/8) != 448/8; new_len++);

	//memcpy(msg, initial_msg, initial_len);
	for (int i = 0; i < initial_len; i++)
		msg[i] = initial_msg[i];
	msg[initial_len] = 0x80; // append the "1" bit; most significant bit is "first"
	for (offset = initial_len + 1; offset < new_len; offset++)
		msg[offset] = 0; // append "0" bits

	// append the len in bits at the end of the buffer.
	to_bytes2(initial_len * 8, msg + new_len);
	// initial_len>>29 == initial_len*8>>32, but avoids overflow.
	to_bytes2(initial_len >> 29, msg + new_len + 4);

	// Process the message in successive 512-bit chunks:
	//for each 512-bit chunk of message:
	for (offset = 0; offset < new_len; offset += (512 / 8)) {

		// break chunk into sixteen 32-bit words w[j], 0 ≤ j ≤ 15
		for (i = 0; i < 16; i++)
			w[i] = to_int32(msg + offset + i * 4);

		// Initialize hash value for this chunk:
		a = h0;
		b = h1;
		c = h2;
		d = h3;

		// Main loop:
		for (i = 0; i < 64; i++) {

			if (i < 16) {
				f = (b & c) | ((~b) & d);
				g = i;
			}
			else if (i < 32) {
				f = (d & b) | ((~d) & c);
				g = (5 * i + 1) % 16;
			}
			else if (i < 48) {
				f = b ^ c ^ d;
				g = (3 * i + 5) % 16;
			}
			else {
				f = c ^ (b | (~d));
				g = (7 * i) % 16;
			}

			temp = d;
			d = c;
			c = b;
			b = b + LEFTROTATE((a + f + k[i] + w[g]), r[i]);
			a = temp;

		}

		// Add this chunk's hash to result so far:
		h0 += a;
		h1 += b;
		h2 += c;
		h3 += d;

	}

	//var char digest[16] := h0 append h1 append h2 append h3 //(Output is in little-endian)
	to_bytes2(h0, digest);
	to_bytes2(h1, digest + 4);
	to_bytes2(h2, digest + 8);
	to_bytes2(h3, digest + 12);
}

__device__ bool compareHash(uint8_t* x1, uint8_t* x2) {
	for (int i = 0; i < 16; i++) {
		if (x1[i] != x2[i])
			return false;
	}
	return true;
}

__device__ void strReverse(char* str, int wordLen) {
	char temp;
	for (int i = 0; i < wordLen / 2; i++)
	{
		temp = str[i];
		str[i] = str[wordLen - i - 1];
		str[wordLen - i - 1] = temp;
	}
}

__device__ void n_to_c(unsigned long long inputNum, int wordLen, char* output) {
	int index = 0;  // Initialize index of result

	while (inputNum > 0)
	{
		output[index++] = inputNum % 26 + 'a';
		inputNum /= 26;
	}

	while (index < wordLen)
		output[index++] = 'a';

	// Reverse the result
	strReverse(output, wordLen);

	output[index] = '\0';
}

__device__ bool generateCombinations(char* word, int wordLen, unsigned long long left, uint8_t* expResult, char* foundWord, uint8_t* md5Temp) {
	int backwards = wordLen - 1;
	for (unsigned long long i = 0; i < left; i++)
	{
		uint8_t md5Result[16];
		md5((uint8_t*)word, wordLen, md5Result, md5Temp);

		if (compareHash(md5Result, expResult)) {
			//printf("fnd %s\n",word);
			for (int i = 0; i < wordLen + 1; i++)
				foundWord[i] = word[i];
			return true;
		}

		if (word[backwards] == 'z')
		{
			while (backwards >= 0)
			{
				if (word[backwards] == 'z')
				{
					word[backwards--] = 'a';
				}
				else
				{
					word[backwards]++;
					break;
				}
			}
			if (backwards < 0) // overflow with ceil()
				return false;

			backwards = wordLen - 1;
		}
		else
		{
			word[backwards]++;
		}
	}
	return false;
}

__global__ void kernel_hash(char* words, int wordLen, unsigned long long offset, unsigned long long eachThread, uint8_t* expResult, char* foundWord, uint8_t* md5Temp) {
	int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int wordIndex = (wordLen + 1) * threadIndex;

	//printf("%llu - %llu\n",(unsigned long long)threadIndex*eachThread, (unsigned long long)threadIndex * eachThread + left - 1);

	n_to_c((unsigned long long)threadIndex * eachThread + offset, wordLen, &words[wordIndex]);
	generateCombinations(&words[wordIndex], wordLen, eachThread, expResult, foundWord, &md5Temp[64 * threadIndex]);
	// todo kill the rest threads if return true

	return;
}

void ErrorCheck(cudaError_t cerr) {
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));
}

int run_md5(int argc, char** argv) {
	if (argc < 6) {
		for (int i = 0; i < argc; i++)
			printf("%s\n", argv[i]);
		printf("wrong args %d. example: ./main 6 `echo -n zzzzzz |md5sum` 200 1024\n", argc);
		return 1;
	}

	int wordLen = atoi(argv[1]);
	char* inputHash = argv[2];
	int blockCount = atoi(argv[4]);
	int threadCount = atoi(argv[5]);

	// converting input chars into uint8_t
	uint8_t expHash[16]; // Expected hash
	char* temp = (char*)malloc(2 * sizeof(char));
	for (int i = 0; i < 16; i++) {
		memcpy(temp, inputHash + i * 2, 2 * sizeof(char));
		expHash[i] = (uint8_t)strtol(temp, NULL, 16);
	}

	char* outputWord = (char*)malloc((wordLen + 1) * sizeof(char));
	outputWord[wordLen] = 'a'; // for kernel output check

	int cudaDevicesCount;
	ErrorCheck(cudaGetDeviceCount(&cudaDevicesCount));
	if (argc > 6 && atoi(argv[6]) < cudaDevicesCount)
		cudaDevicesCount = atoi(argv[6]);

	int debugText = 1;
	if (argc > 7)
		debugText = atoi(argv[7]);

	unsigned long long eachDevice = (unsigned long long)ceil(pow(26, wordLen) / cudaDevicesCount);
	unsigned long long eachThread = (unsigned long long)ceil(eachDevice / (1.0 * blockCount * threadCount));

	uint8_t** expHashP = (uint8_t**)malloc(cudaDevicesCount * sizeof(uint8_t*));
	char** outputWordP = (char**)malloc(cudaDevicesCount * sizeof(char*));
	char** wordsP = (char**)malloc(cudaDevicesCount * sizeof(char*));
	uint8_t** md5TempP = (uint8_t**)malloc(cudaDevicesCount * sizeof(uint8_t*));

	if(debugText)
		printf("Running %dx kernel, with %d blocks and %d threads, each thread in grid will solve %llu combinations.\n", cudaDevicesCount, blockCount, threadCount, eachThread);

	auto start_time = std::chrono::high_resolution_clock::now();

	for (int offset = 0; offset < cudaDevicesCount; offset++) {
		cudaSetDevice(offset);

		ErrorCheck(cudaMalloc(&expHashP[offset], 16 * sizeof(uint8_t)));
		ErrorCheck(cudaMemcpy(expHashP[offset], expHash, 16 * sizeof(uint8_t), cudaMemcpyHostToDevice));

		// (wordLen+1) because last is '\0'
		ErrorCheck(cudaMalloc(&wordsP[offset], (wordLen + 1) * threadCount * blockCount));

		ErrorCheck(cudaMalloc(&outputWordP[offset], (wordLen + 1) * sizeof(char)));
		ErrorCheck(cudaMemcpy(outputWordP[offset], outputWord, (wordLen + 1) * sizeof(char), cudaMemcpyHostToDevice));

		size_t md5TempLen = 64; /* seems to be 64 every time
		for (md5TempLen = wordLen + 1; md5TempLen % (512/8) != 448/8; md5TempLen++);
		md5TempLen = md5TempLen+8; */
		//printf("whole: %lu, piece: %lu \n",threadCount*md5TempLen,md5TempLen);

		// temporary array inside of md5 alg
		ErrorCheck(cudaMalloc(&md5TempP[offset], blockCount * threadCount * md5TempLen));

		kernel_hash << <blockCount, threadCount >> > (wordsP[offset], wordLen, offset * eachDevice, eachThread, expHashP[offset], outputWordP[offset], md5TempP[offset]);
	}

	int foundHash = 0;
	for (int offset = 0; offset < cudaDevicesCount; offset++) {
		cudaSetDevice(offset);

		if (!foundHash) {
			// Copy data from GPU device to PC
			ErrorCheck(cudaMemcpy(outputWord, outputWordP[offset], (wordLen + 1) * sizeof(char), cudaMemcpyDeviceToHost));

			if (outputWord[wordLen] == '\0') {
				foundHash = 1;
			}
		}

		// Free memory
		cudaFree(outputWordP[offset]);
		cudaFree(md5TempP[offset]);
		cudaFree(expHashP[offset]);
		cudaFree(wordsP[offset]);
	}

	auto time = std::chrono::high_resolution_clock::now() - start_time;

	// waiting for console output
	cudaDeviceSynchronize();

	if (debugText) {
		if (foundHash)
			printf("found %s\n", outputWord);
		else
			printf("did not found the hash\n");
	}

	std::cout << wordLen << "; " << time / std::chrono::milliseconds(1) / 1000.0 << ";" << std::endl;
	return !foundHash; // when OK, 0 is expected
}
