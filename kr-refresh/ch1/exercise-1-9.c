#include <stdio.h>

/* count blanks, tabs, and newlines */
main()
{
    int c;
    while ((c = getchar()) != EOF) {
        putchar(c);
        if (c == ' ') {
            while ((c = getchar()) == ' ')
                ;
            putchar(c);
        }
    }
}
