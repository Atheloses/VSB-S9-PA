#include <stdio.h>

// Function prototype from .cu file
int run_md5(int argc, char** argv);

int main(int argc, char** argv)
{
	return run_md5(argc, argv);
}

