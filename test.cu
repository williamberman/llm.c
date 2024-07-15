#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cuda/ptx>

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// #define DIM 16
// #define VOCAB_SIZE 32
// #define B 2
// #define T 2

#define DIM 1600 // 4096 // 1600
#define VOCAB_SIZE 65536
#define B 4 // 2 // 4
#define T 1024

template<typename Type>
__global__ void embedding_backward_kernel(
    Type *dweight, // VOCAB_SIZE, DIM
    const Type *dout, // B, T, DIM
    const uint16_t *input_tokens, // B, T
    const uint16_t *tokens_for_thread_blocks, // n_unique_tokens
    const size_t n_splits
) {
    extern __shared__ Type smem[]; // X, DIM
    __shared__ uint64_t barrier;
    int smem_copies = 0;
    uint16_t thread_block_token = tokens_for_thread_blocks[blockIdx.x / n_splits];

    size_t dim = DIM / n_splits;
    assert(dim % 16 == 0);
    size_t offset = (blockIdx.x % n_splits) * dim;

    if (threadIdx.x == 0) {
        cuda::ptx::mbarrier_init(&barrier, 1);

        cuda::ptx::fence_proxy_async(cuda::ptx::space_cluster);

        for (int token_idx = 0; token_idx < B*T; ++token_idx) {
            if (input_tokens[token_idx] == thread_block_token) {
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_cluster, 
                    cuda::ptx::space_global, 
                    smem + smem_copies * dim, 
                    dout + token_idx * DIM + offset, 
                    dim * sizeof(Type), 
                    &barrier
                );
                ++smem_copies;
            }
        }

        cuda::ptx::mbarrier_arrive_expect_tx(
            cuda::ptx::sem_release, 
            cuda::ptx::scope_cta, 
            cuda::ptx::space_shared, 
            &barrier, 
            dim * sizeof(Type) * smem_copies
        );
    }

    __syncthreads();

    bool complete = false;
    while (!complete) {
        complete = cuda::ptx::mbarrier_try_wait_parity(&barrier, 0);
    }

    if (threadIdx.x == 0) {
        for (int i = 0; i < smem_copies; ++i) {
            cuda::ptx::cp_reduce_async_bulk(
                cuda::ptx::space_global, 
                cuda::ptx::space_shared, 
                cuda::ptx::op_add, 
                dweight + thread_block_token * DIM + offset, 
                smem + i * dim, 
                dim * sizeof(Type)
            );
        }
    }
}

template<typename Type>
void embedding_backward(
    Type *dweight,
    const Type *dout, 
    const uint16_t *input_tokens_device,
    const uint16_t *input_tokens_host // B, T
) {
    int n_unique_tokens = 0;
    uint64_t unique_tokens[1024];
    uint16_t tokens_for_thread_blocks_host[B*T];

    for (int token_idx = 0; token_idx < B*T; ++token_idx) {
        uint16_t token = input_tokens_host[token_idx];
        bool token_is_not_found = (unique_tokens[token / 64] & (1ULL << (token % 64))) == 0;

        if (token_is_not_found) {
            tokens_for_thread_blocks_host[n_unique_tokens] = token;
            n_unique_tokens += 1;
            unique_tokens[token / 64] |= (1ULL << (token % 64));
        }
    }

    uint16_t *tokens_for_thread_blocks_device;
    cudaCheck(cudaMalloc(&tokens_for_thread_blocks_device, n_unique_tokens * sizeof(uint16_t)));
    cudaCheck(cudaMemcpy(tokens_for_thread_blocks_device, tokens_for_thread_blocks_host, n_unique_tokens * sizeof(uint16_t), cudaMemcpyHostToDevice));

    int max_dynamic_shared_memory = 230000;
    size_t n_splits = 100; // 160; // 1; // 160;

    cudaFuncSetAttribute(embedding_backward_kernel<Type>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_shared_memory);

    embedding_backward_kernel<Type><<<n_unique_tokens*n_splits, 128, max_dynamic_shared_memory>>>(
        dweight,
        dout, 
        input_tokens_device, 
        tokens_for_thread_blocks_device,
        n_splits
    );
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaFree(tokens_for_thread_blocks_device));
}

int main() {
    // uint16_t *input_tokens_host = new uint16_t[4]; // [B*T];
    // input_tokens_host[0] = 0;
    // input_tokens_host[1] = 0;
    // input_tokens_host[2] = 2;
    // input_tokens_host[3] = 3;

    uint16_t *input_tokens_host = new uint16_t[B*T]; // [B*T];
    for (int b = 0; b < 2; ++b) {
        for (int t = 0; t < T; ++t) {
            input_tokens_host[b*T + t] = 0;
        }
    }
    for (int b = 2; b < 4; ++b) {
        for (int t = 0; t < T; ++t) {
            input_tokens_host[b*T + t] = 1;
        }
    }

    float *dout_device = new float[B*T*DIM];
    for (int i = 0; i < B*T*DIM; ++i) {
        dout_device[i] = 1.0;
    }

    float *dweight;
    float *dout;
    uint16_t *input_tokens_device;

    cudaCheck(cudaMalloc(&dweight, DIM * VOCAB_SIZE * sizeof(float)));

    cudaMalloc(&dout, B * T * DIM * sizeof(float));
    cudaCheck(cudaMemcpy(dout, dout_device, B * T * DIM * sizeof(float), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&input_tokens_device, B*T*sizeof(uint16_t)));
    cudaCheck(cudaMemcpy(input_tokens_device, input_tokens_host, B*T*sizeof(uint16_t), cudaMemcpyHostToDevice));

    embedding_backward(
        dweight,
        dout, 
        input_tokens_device,
        input_tokens_host
    );

    float *dweight_host = new float[DIM * VOCAB_SIZE];
    cudaCheck(cudaMemcpy(dweight_host, dweight, DIM * VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // for (int n = 0; n < B*T; ++n) {
    //     for (int i = 0; i < DIM; ++i) {
    //         printf("%f ", dweight_host[n*DIM + i]);
    //     }
    //     printf("\n********************\n");
    // }

    // for (int i = 0; i < DIM*VOCAB_SIZE; ++i) {
    //     float expected;
    //     if (i < DIM) {
    //         expected = 2.0;
    //     } else if (i < 2*DIM) {
    //         expected = 0.0;
    //     } else if (i < 4*DIM) {
    //         expected = 1.0;
    //     } else {
    //         expected = 0.0;
    //     }

    //     if (dweight_host[i] != expected) {
    //         printf("dweight_host[%d] == %f, expected %f\n", i, dweight_host[i], expected);
    //         assert(false);
    //     }
    // }

    for (int i = 0; i < DIM*VOCAB_SIZE; ++i) {
        float expected;
        if (i < 2*DIM) {
            expected = B*T / 2;
        } else {
            expected = 0.0;
        }

        if (dweight_host[i] != expected) {
            printf("dweight_host[%d] == %f, expected %f\n", i, dweight_host[i], expected);
            assert(false);
        }
    }

    printf("passed!\n");

    delete[] input_tokens_host;

    return 0;
}
