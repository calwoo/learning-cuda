#include <stdio.h>

/* count blanks, tabs, and newlines */
main()
{
    int c, b, t, nl;

    b = 0;
    t = 0;
    nl = 0;
    while ((c = getchar()) != EOF) {
        if (c == '\n')
            ++nl;
        else if (c == '\t')
            ++t;
        else if (c == ' ')
            ++b;
    }
    printf("%d %d %d", b, t, nl);
}
