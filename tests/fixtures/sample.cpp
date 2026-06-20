// C++ fixture exercising the constructs the tree-sitter-cpp extractor must
// capture: namespace, function template, class with ctor/dtor/inline method,
// out-of-line method, operator overload, and a free function with a lambda.
#include <cstring>
#include <cstdlib>

namespace net {

template <typename T>
T clamp(T v, T lo, T hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

class Buffer {
public:
    Buffer(const char *s) {            // constructor
        strcpy(buf_, s);               // CWE-120: unbounded copy into fixed buffer
    }
    ~Buffer() {}                       // destructor
    int size() const { return n_; }    // inline member
    Buffer &operator=(const Buffer &o); // declaration only -> NOT captured
    void copy(const char *s);           // declaration only -> NOT captured

private:
    char buf_[16];
    int n_;
};

void Buffer::copy(const char *s) {     // out-of-line method definition
    strcpy(buf_, s);
}

Buffer &Buffer::operator=(const Buffer &o) {  // out-of-line operator overload
    std::memmove(buf_, o.buf_, sizeof(buf_));
    return *this;
}

} // namespace net

int run(const char *cmd) {
    auto wrap = [](const char *c) { return system(c); };  // lambda inside run()
    return wrap(cmd);
}
