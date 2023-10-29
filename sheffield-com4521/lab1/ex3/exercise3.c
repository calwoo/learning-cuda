/* Hello World program */
#include <stdio.h>
#include "random.h"

#define NUM_VALUES 250

long long value[NUM_VALUES];

int main()
{
    float sum = 0;
    float min, max;
    float average = 0;
    unsigned char i = 0;
    init_random();

    for (i = 0; i < NUM_VALUES; i++) {
        value[i] = random_float();
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

    printf("SUM=%.0f\n", sum);
    printf("AVERAGE=%.0f\n", average);
    printf("MIN=%.0f\n", min);
    printf("MAX=%.0f\n", max);
    return 0;
}
