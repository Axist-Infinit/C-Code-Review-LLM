/* A small fixture exercising several extraction edge cases. */

struct point {
    int x;
    int y;
};

static const char *names[] = {
    "alice", "bob"
};

// multi-line signature + return type on its own line
int
add(int a, int b)
{
    return a + b;
}

static void greet(const char *who) {
    char buf[16];
    strcpy(buf, who);
}
