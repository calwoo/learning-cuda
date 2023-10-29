/* Hello World program */
#include <stdio.h>
#include "random.h"

#define NUM_VALUES 250

int value[NUM_VALUES];

int main()
{
    unsigned int sum = 0;
    int min, max;
    float average = 0;
    unsigned char i = 0;
    init_random();

    for (i = 0; i < NUM_VALUES; i++) {
        value[i] = random_ushort();
        // printf("%d %d\n", i, value[i]);
    }

    for (i = 0; i < NUM_VALUES; i++) {
        sum += value[i];
    }
    average = ((float) sum) / NUM_VALUES;

    // normalize
    min = value[0];
    max = value[0];
    for (i = 0; i < NUM_VALUES; i++) {
        min = (value[i] < min) ? value[i] : min;
        max = (value[i] > max) ? value[i] : max;
    }

    printf("SUM=%d\n", sum);
    printf("AVERAGE=%.0f\n", average);
    printf("MIN=%d\n", min);
    printf("MAX=%d\n", max);
    return 0;
}
