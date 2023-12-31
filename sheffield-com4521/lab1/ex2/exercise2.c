/* Hello World program */
#include <stdio.h>
#include "random.h"

#define NUM_VALUES 250

long long value[NUM_VALUES];

int main()
{
    unsigned long long sum = 0;
    long long min, max;
    unsigned long long average = 0;
    unsigned char i = 0;
    init_random();

    for (i = 0; i < NUM_VALUES; i++) {
        value[i] = random_uint();
        // printf("%d %d\n", i, value[i]);
    }

    for (i = 0; i < NUM_VALUES; i++) {
        sum += value[i];
    }
    average = sum / NUM_VALUES;

    // normalize
    for (i = 0; i < NUM_VALUES; i++) {
        value[i] -= average;
    }

    min = value[0];
    max = value[0];
    for (i = 0; i < NUM_VALUES; i++) {
        min = (value[i] < min) ? value[i] : min;
        max = (value[i] > max) ? value[i] : max;
    }

    printf("SUM=%llu\n", sum);
    printf("AVERAGE=%u\n", average);
    printf("MIN=%lld\n", min);
    printf("MAX=%lld\n", max);
    return 0;
}
