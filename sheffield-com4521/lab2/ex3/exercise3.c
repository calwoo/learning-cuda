#include <stdio.h>
#include <stdlib.h>
#include "linked_list.h"

#define NUM_STUDENTS 4

struct student {
	char *forename;
	char *surname;
	float average_module_mark;
};

void print_student(const struct student *s);

void main() {
	FILE *f = NULL;
	f = fopen("students2.bin", "rb"); //read and binary flags
	if (f == NULL) {
		fprintf(stderr, "Error: Could not find students.bin file \n");
		exit(1);
	}

	print_callback = (void (*) (void*)) &print_student;

	llitem* students = NULL;
	llitem* end = NULL;

	int i;

	// read data
	unsigned int strlen;
	while (fread(&strlen, sizeof(unsigned int), 1, f) == 1) {
		struct student *s;
		s = (struct student*) malloc(sizeof(struct student));
		// read forename
		s->forename = (char*) malloc(sizeof(char) * strlen);
		fread(s->forename, sizeof(char), strlen, f);
		// read surname
		fread(&strlen, sizeof(unsigned int), 1, f);
		s->surname = (char*) malloc(sizeof(char) * strlen);
		fread(s->surname, sizeof(char), strlen, f);
		// read average mark
		fread(&s->average_module_mark, sizeof(float), 1, f);

		// add to linked list
		if (end == NULL) {
			end = create_linked_list();
			students = end;
		} else {
			end = add_to_linked_list(end);
		}
		end->record = (void*) s;
	}
	fclose(f);

	print_items(students);

	// free memory
	llitem* tmp;
	while (students != NULL) {
		tmp = students->next;
		free(((struct student*) students->record)->forename);
		free(((struct student*) students->record)->surname);
		free(students->record);
		students = tmp;
	}
	free_linked_list(students);
}

void print_student(const struct student *s) {
	printf("Student:\n");
	printf("\tForename: %s\n", s->forename);
	printf("\tSurname: %s\n", s->surname);
	printf("\tAverage Module Mark: %.2f\n", s->average_module_mark);
}

