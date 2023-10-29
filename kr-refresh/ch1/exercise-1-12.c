#include <stdio.h>
#define IN 1
#define OUT 0

main() {
    int c, state;
  
  	state = IN;
    while ((c = getchar()) != EOF) {
      	if (c == ' ' || c == '\n')
        	state = OUT;
     	else if (state == OUT) {
            printf("\n");
      		state = IN;
        }

      	if (state == IN)
        	putchar(c);
    }
}
