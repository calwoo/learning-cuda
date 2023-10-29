#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>

#define BUFFER_SIZE 32

int readLine(char buffer[]);

int main()
{
    float in_value, sum;
	char buffer[BUFFER_SIZE];
	char command[4];
    sum = 0;

	printf("Welcome to basic COM4521 calculator\nEnter command: ");

    while (readLine(buffer)){
		if (!(isalpha(buffer[0]) && isalpha(buffer[1]) && isalpha(buffer[2]) && (buffer[3] == ' '))) {
			fprintf(stderr, "Incorrect command format\n");
			continue;
		}
		sscanf(buffer, "%s %f", command, &in_value);

		if (strcmp("add", command) == 0) {
			sum += in_value;
		} else if (strcmp("sub", command) == 0) {
			sum -= in_value;
		} else if (strcmp("mul", command) == 0) {
			sum *= in_value;
		} else if (strcmp("div", command) == 0) {
			sum /= in_value;
		} else if (strncmp("ad", command, 2) == 0) {
			printf("Did you mean add?\n");
		} else if (strncmp("su", command, 2) == 0) {
			printf("Did you mean sub?\n");
		} else if (strncmp("mu", command, 2) == 0) {
			printf("Did you mean mul?\n");
		} else if (strncmp("di", command, 2) == 0) {
			printf("Did you mean div?\n");
		} else {
			printf("Unknown command\n");
		}

		printf("\tSum is %f\n", sum);
		printf("Enter next command: ");
	}

    return 0;
}

int readLine(char buffer[]) {
	int i = 0;
	char c = 0;
	while ((c = getchar()) != '\n') {
		buffer[i] = c;
		i++;
		if (i >= BUFFER_SIZE) {
			fprintf(stderr, "too many chars, buffer overflow");
			exit(1);
		}
	}
	buffer[i] = '\0';

	if (strcmp(buffer, "exit") == 0)
		return 0;
	else
		return 1;
}