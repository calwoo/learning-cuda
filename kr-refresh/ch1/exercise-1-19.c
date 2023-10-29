#include <stdio.h>
#define MAXLINE 1000

void reverse(char s[]);

int main()
{
    char test[MAXLINE] = "hello world";
    reverse(test);

    printf("%s\n", test);
    return 0;
}

void reverse(char s[])
{
    char tmp;

    // get length of string
    int i = 0;
    while (s[i] != '\0')
        i++;
    
    // reverse string
    i--;
    int j = 0;
    while (j < i) {
        tmp = s[j];
        s[j] = s[i];
        s[i] = tmp;
        i--;
        j++;
    }
}
