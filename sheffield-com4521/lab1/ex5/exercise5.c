#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>

#define BUFFER_SIZE 32

int readLine(FILE *f, char buffer[]);

int main()
{
	FILE *commands;
	commands = fopen("commands.calc", "r");

    float in_value, sum;
	char buffer[BUFFER_SIZE];
	char command[4];
    sum = 0;

    while (readLine(commands, buffer)){
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
		} else {
			printf("Unknown command\n");
		}
	}
	printf("Sum is %f\n", sum);
	
	fclose(commands);
    return 0;
}

int readLine(FILE *f, char buffer[]) {
	int i = 0;
	char c;
	while ((c = getc(f)) != '\n') {
		if (c == EOF)
			return 0;
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