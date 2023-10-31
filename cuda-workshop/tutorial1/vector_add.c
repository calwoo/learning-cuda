#define N 10000000

void vector_add(float *out, float *a, float *b, int n) {
    for (int i = 0; i < N; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *b, *out;

    // allocate
    a = (float*) malloc(sizeof(float) * N);
    b = (float*) malloc(sizeof(float) * N);
    out = (float*) malloc(sizeof(float) * N);

    // initialize
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    vector_add(out, a, b, N);
}
