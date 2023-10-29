#include <stdio.h>
#include <stdlib.h>

#define NUM_STUDENTS 4

struct student {
	char *forename;
	char *surname;
	float average_module_mark;
};

void print_student(const struct student *s);

void main() {
	struct student *students;
	students = (struct student*) malloc(sizeof(struct student) * NUM_STUDENTS);

	int i;

	FILE *f = NULL;
	f = fopen("students2.bin", "rb"); //read and binary flags
	if (f == NULL) {
		fprintf(stderr, "Error: Could not find students.bin file \n");
		exit(1);
	}

	// read data
	for (i = 0; i< NUM_STUDENTS; i++) {
		unsigned int strlen;
		// read forename
		fread(&strlen, sizeof(unsigned int), 1, f);
		students[i].forename = (char*) malloc(sizeof(char) * strlen);
		fread(students[i].forename, sizeof(char), strlen, f);
		// read surname
		fread(&strlen, sizeof(unsigned int), 1, f);
		students[i].surname = (char*) malloc(sizeof(char) * strlen);
		fread(students[i].surname, sizeof(char), strlen, f);
		// read average mark
		fread(&students[i].average_module_mark, sizeof(float), 1, f);
	}
	fclose(f);

	for (i = 0; i < NUM_STUDENTS; i++) {
		print_student(&students[i]);
	}

	// free memory
	for (i = 0; i < NUM_STUDENTS; i++) {
		free(students[i].forename);
		free(students[i].surname);
	}
	free(students);
}

void print_student(const struct student *s) {
	printf("Student:\n");
	printf("\tForename: %s\n", s->forename);
	printf("\tSurname: %s\n", s->surname);
	printf("\tAverage Module Mark: %.2f\n", s->average_module_mark);
}

