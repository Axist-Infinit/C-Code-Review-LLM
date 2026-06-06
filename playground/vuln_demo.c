\
#include <stdio.h>
#include <string.h>

void greet() {
    char name[16];
    printf("Enter your name: ");
    gets(name);                // UNSAFE: no bounds check
    char msg[32];
    sprintf(msg, "Hello %s!", name);  // UNSAFE: sprintf
    printf("%s\n", msg);
}

void copy_user(char *src) {
    char dst[8];
    strcpy(dst, src);          // UNSAFE: strcpy
    printf("%s\n", dst);
}

int main(int argc, char **argv) {
    if (argc > 1) {
        copy_user(argv[1]);
    }
    greet();
    return 0;
}
