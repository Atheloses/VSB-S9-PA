#include <stdio.h>

// Function prototype from .cu file
int run_gof(int argc, char **argv);

int main(int argc, char **argv)
{
	return run_gof(argc, argv);
}

