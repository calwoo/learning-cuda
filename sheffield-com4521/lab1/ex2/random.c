#include <stdlib.h>
#include "random.h"


void init_random() {
	srand(RAND_SEED);
}

unsigned short random_ushort() {
	return (unsigned short) rand();
}
