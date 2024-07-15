#include <stdio.h>

#include <cuda/ptx>

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

#define numel 16

__global__ void kernel(float *inp) {
    __shared__ float smem[2*numel];
    __shared__ uint64_t barrier;
    bool complete_0;
    bool complete_1;

    if (threadIdx.x == 0) {
        cuda::ptx::mbarrier_init(&barrier, blockDim.x*1);

        complete_0 = cuda::ptx::mbarrier_try_wait_parity(&barrier, 0);
        complete_1 = cuda::ptx::mbarrier_try_wait_parity(&barrier, 1);
        printf("complete_0: %d, complete_1: %d\n", complete_0, complete_1);

        cuda::ptx::fence_proxy_async(cuda::ptx::space_cluster);
        cuda::ptx::cp_async_bulk(cuda::ptx::space_cluster, cuda::ptx::space_global, smem, inp, 4*numel, &barrier);
        cuda::ptx::cp_async_bulk(cuda::ptx::space_cluster, cuda::ptx::space_global, smem+numel, inp, 4*numel, &barrier);

        // complete_0 = cuda::ptx::mbarrier_try_wait_parity(&barrier, 0);
        // complete_1 = cuda::ptx::mbarrier_try_wait_parity(&barrier, 1);
        // printf("complete_0: %d, complete_1: %d\n", complete_0, complete_1);
    }

    __syncthreads();

    // cuda::ptx::fence_proxy_async();
    complete_0 = cuda::ptx::mbarrier_test_wait_parity(&barrier, 0);
    complete_1 = cuda::ptx::mbarrier_test_wait_parity(&barrier, 1);
    if (threadIdx.x == 0) {
        printf("complete_0: %d, complete_1: %d\n", complete_0, complete_1);
    }

    cuda::ptx::mbarrier_arrive(&barrier, 1);

    __syncthreads();

    cuda::ptx::fence_proxy_async();
    complete_0 = cuda::ptx::mbarrier_test_wait_parity(&barrier, 0);
    complete_1 = cuda::ptx::mbarrier_test_wait_parity(&barrier, 1);
    if (threadIdx.x == 0) {
        printf("complete_0: %d, complete_1: %d\n", complete_0, complete_1);
    }

    bool complete = false; 
    while (!complete) {
        complete = cuda::ptx::mbarrier_test_wait_parity(&barrier, 0);
    }

    if (threadIdx.x == 0) {
        printf("complete: %d\n", complete);
        printf("smem: ");
        for (int i = 0; i < numel; ++i) {
            printf("%f ", smem[i]);
        }
        printf("\n");
    }
}

int main() {
    float inp_host[numel];
    for (int i = 0; i < numel; ++i) {
        inp_host[i] = i;
    }

    float *inp_device;
    cudaCheck(cudaMallocHost(&inp_device, numel * sizeof(float)));
    cudaCheck(cudaMemcpy(inp_device, inp_host, numel * sizeof(float), cudaMemcpyHostToDevice));

    int max_dynamic_shared_memory = 230000;
    int max_threads_per_block = 1024;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_shared_memory);

    kernel<<<1, 1024>>>(inp_device);
    // kernel<<<1, max_threads_per_block, max_dynamic_shared_memory>>>(inp_device);

    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());

    return 0;
}