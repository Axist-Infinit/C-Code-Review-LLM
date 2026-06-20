// C++ demo for the scanner: class methods, an operator overload, a template,
// and a free function — each carrying a representative vulnerability.
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>

class Session {
public:
    explicit Session(const char *user) {
        std::strcpy(name_, user);          // CWE-120: unbounded copy into name_[32]
    }

    void log(const char *fmt) {
        printf(fmt);                        // CWE-134: non-literal format string
    }

    int shell(const std::string &cmd) {
        return system(cmd.c_str());         // CWE-78: command injection
    }

private:
    char name_[32];
};

template <typename T>
T *as_raw(void *p) {
    return reinterpret_cast<T *>(p);        // CWE-704: unchecked type reinterpretation
}

int main(int argc, char **argv) {
    if (argc > 1) {
        Session s(argv[1]);
        s.shell(argv[1]);
    }
    return 0;
}
