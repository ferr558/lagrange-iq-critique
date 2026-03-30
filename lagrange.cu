#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define MAX_POINTS   10
#define NUM_K_VALUES 1024
#define NUM_X_EVAL   512
#define OUTPUT_CSV   "output.csv"

static void HandleCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(err) HandleCudaError((err), __FILE__, __LINE__)

__device__ float lagrange_basis(float x, int i, float *xs, int n) {

    float result = 1.0f;

    for (int j = 0; j < n; j++) {
        if (j != i) {
            result *= (x - xs[j]) / (xs[i] - xs[j]);
        }
    }

    return result;

}

__device__ float lagrange_eval(float x, float *xs, float *ys, int n) {

    float result = 0.0f;

    for (int i = 0; i < n; i++) {
        result += ys[i] * lagrange_basis(x, i, xs, n);
    }

    return result;

}

__device__ float perturbation(float x, float *xs, int n) {

    float result = 1.0f;

    for (int i = 0; i < n; i++) {
        result *= (x - xs[i]);
    }

    return result;

}

__global__ void lagrange_kernel(float *xs, float *ys, int n, float x_start, float x_end, float *out, unsigned long long seed) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_K_VALUES) return;

    curandState state;
    curand_init(seed, tid, 0, &state);
    float k = curand_uniform(&state) * 10.0f - 5.0f;  // k in [-5, 5]

    for (int ix = 0; ix < NUM_X_EVAL; ix++) {

        float x = x_start + (x_end - x_start) * ix / (NUM_X_EVAL - 1);
        float val = lagrange_eval(x, xs, ys, n) + k * perturbation(x, xs, n);
        out[tid * NUM_X_EVAL + ix] = val;

    }

}

int main() {

    int n;

    printf("Quanti termini ha la sequenza? ");
    scanf("%d", &n);

    if (n < 2 || n > MAX_POINTS) {

        fprintf(stderr, "Inserisci tra 2 e %d punti.\n", MAX_POINTS);
        return EXIT_FAILURE;

    }

    float ys[MAX_POINTS];
    float xs[MAX_POINTS];

    printf("Inserisci i %d valori della sequenza:\n", n);

    for (int i = 0; i < n; i++) {

        xs[i] = (float)i;
        printf("  y[%d] = ", i);
        scanf("%f", &ys[i]);

    }

    float x_start = -0.5f;
    float x_end   = (float)n + 0.5f;
    float *d_xs, *d_ys, *d_out;

    CUDA_CHECK(cudaMalloc(&d_xs,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ys,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, NUM_K_VALUES * NUM_X_EVAL * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, n * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (NUM_K_VALUES + threads - 1) / threads;
    unsigned long long seed = 42ULL;

    lagrange_kernel<<<blocks, threads>>>(d_xs, d_ys, n, x_start, x_end, d_out, seed);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_out = (float*)malloc(NUM_K_VALUES * NUM_X_EVAL * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_out, d_out, NUM_K_VALUES * NUM_X_EVAL * sizeof(float), cudaMemcpyDeviceToHost));

    FILE *fout = fopen(OUTPUT_CSV, "w");

    fprintf(fout, "x,y_min,y_max\n");

    for (int ix = 0; ix < NUM_X_EVAL; ix++) {

        float x   = x_start + (x_end - x_start) * ix / (NUM_X_EVAL - 1);
        float ymin =  1e30f;
        float ymax = -1e30f;

        for (int k = 0; k < NUM_K_VALUES; k++) {

            float val = h_out[k * NUM_X_EVAL + ix];

            if (val < ymin) ymin = val;
            if (val > ymax) ymax = val;

        }

        fprintf(fout, "%f,%f,%f\n", x, ymin, ymax);

    }

    fclose(fout);

    FILE *fpts = fopen("points.csv", "w");

    fprintf(fpts, "n,%d\n", n);

    for (int i = 0; i < n; i++)
        fprintf(fpts, "%f,%f\n", xs[i], ys[i]);

    fprintf(fpts, "k,-3.0,0.0,3.0\n");
    fclose(fpts);

    free(h_out);
    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_out);

    printf("Scritto output.csv e points.csv\n");
    printf("Esegui: python3 plot.py\n");
    return 0;
    
}