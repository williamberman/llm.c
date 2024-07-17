/*
The GPT-2 Encoder, which combines two encodings: token and position
In the forward pass, both encodings are added together
In the backward pass, the gradients flow to both, handled by different kernels
*/
#include <assert.h>
#include <stdint.h>
#include <utility>              // std::pair
#include <vector>
#include <algorithm>
#include <unordered_map>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

#include <unordered_set>
#include <assert.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda/ptx>


// ----------------------------------------------------------------------------
// CUDA kernels

__global__ void encoder_forward_kernel3(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;
    if (idx >= N) { return; }

    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;

    int ix = inp[b * T + t];

    floatX* out_btc = out + b * T * C + t * C + c;
    const floatX* wte_ix = wte + ix * C + c;
    const floatX* wpe_tc = wpe + t * C + c;

    x128 packed_out;
    x128 wte128 = load128cs(wte_ix);
    x128 wpe128 = load128cs(wpe_tc);
    for (int k = 0; k < x128::size; k++) {
        packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
    }
    store128(out_btc, packed_out);
}

template <int BLOCK_SIZE=256>
__global__ void wte_backward_kernel(floatX* dwte,
                                    const int4* bucket_info, const int* workload_indices, const floatX* dout, const int* inp,
                                    unsigned int seed, int B, int T, int C) {
    // In order to be deterministic, we preprocess the inputs on the cpu into "buckets"
    // Each bucket corresponds to (WARP_SIZE * x128::size) channels for a single vocabulary token
    // Each thread handles x128::size channels, e.g. 256 per warp for BF16
    // Each block handles (BLOCK_SIZE / WARP_SIZE) elements in a single bucket in parallel
    // If a bucket has less than 8 elements, some warps will return immediately
    // If a bucket has more than 8 elements, we will loop over all of them
    // The buckets are sorted on the CPU so the largest buckets start 1st
    int bucket = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int c_per_warp = WARP_SIZE * x128::size;

    int bucket_start_idx = bucket_info[bucket].x;
    int bucket_size = bucket_info[bucket].y;
    int bucket_ix = bucket_info[bucket].z;
    int c = bucket_info[bucket].w * c_per_warp + (lane_id * x128::size);

    // Each thread handles "x128::size" channels, so at fp8, each warp would handle 512 channels
    // If C is not a multiple of this (e.g. 768), some buckets/c_groups cannot use the entire warp
    if (c >= C) { return; }
    // Exit early if this is a small bucket and this warp doesn't have any items to process
    if (warp_id >= bucket_size) { return; }

    float accum[x128::size] = {0.0f};
    __shared__ float accum_shared[x128::size * BLOCK_SIZE];

    for(int item = warp_id; item < bucket_size; item += BLOCK_SIZE/WARP_SIZE) {
        int bt = workload_indices[bucket_start_idx + item];

        const floatX* dout_btc = dout + bt * C + c;
        x128 packed_inp1 = load128cs(dout_btc);
        for (int k = 0; k < packed_inp1.size; k++) {
            accum[k] += (float)packed_inp1[k];
        }
    }

    if (warp_id != 0) {
        // we accumulate into warp 0, so only the other warps need to write to shared memory
        for (int k = 0; k < x128::size; k++) {
            accum_shared[threadIdx.x + k * BLOCK_SIZE] = accum[k];
        }
        return; // only warp 0 is needed after writing to shared memory
    }

    // Read dwte for warp 0 even if other warps are not finished yet to maximise latency tolerance
    floatX* dwte_ix = dwte + bucket_ix * C + c;
    x128 packed_in_out = load128(dwte_ix);

    // note: threads which have returned are considered synchronised by CUDA so no risk of deadlock
    __syncthreads();

    // Accumulate into warp 0's registers by reading the values of the other warps in shared memory
    for (int i = threadIdx.x+WARP_SIZE; i < min(BLOCK_SIZE, bucket_size*WARP_SIZE); i += WARP_SIZE) {
        for (int k = 0; k < x128::size; k++) {
            accum[k] += accum_shared[i + k * BLOCK_SIZE];
        }
    }

    // Add the result to dwte and write back to global memory (read-modify-write)
    for (unsigned int k = 0; k < x128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16
        // The seed is deterministic and unique for each parameter to guarantee we have determinism AND
        // to avoid **potential** issues with positionX int SquirrelNoise5 argument overflowing which is UB
        // and that somehow messing the quality of random numbers
        stochastic_rounding(accum[k] + (float)packed_in_out[k], &packed_in_out[k], seed + bucket * WARP_SIZE + threadIdx.x + k);
    }
    store128(dwte_ix, packed_in_out);
}

__global__ void wpe_backward_kernel(floatX* dwpe,
                                    const floatX* dout, const int* inp,
                                    int B, int T, int C, unsigned int seed) {
    // Each thread handles x128::size "channel positions", e.g. 256 per warp for BF16
    // For gpt2-124M BF16, C=768 and T=1024, so 3 warps per channel and 3072 warps in total
    // For each "channel position" we sum the gradients for every batch at that C/T element
    // This way each dwte element is only updated once, and the kernel is fully deterministic!
    // The previous kernel was not deterministic, as batches were aggregated with atomicAdd
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= T * C) { return; }

    // if C is not a multiple of WARP_SIZE*x128::size, it's OK for some warps to handle multiple t
    int t = idx / C;
    int c = idx % C;
    float accum[x128::size] = {0.0f};

    for (int b = 0; b < B; b++) {
        x128 packed_dout = load128cs(dout + (b * T * C) + (t * C) + c); // will never be read again
        for (int k = 0; k < x128::size; k++) {
            accum[k] += (float)packed_dout[k];
        }
    }

    floatX* dwpe_tc = dwpe + (t * C) + c;
    x128 packed_dwpe = load128(dwpe_tc);
    for (unsigned int k = 0; k < x128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16
        // The seed is deterministic and unique for each parameter to guarantee we have determinism AND
        // to avoid **potential** issues with positionX int SquirrelNoise5 argument overflowing which is UB
        // and that somehow messing the quality of random numbers
        stochastic_rounding(accum[k] + (float)packed_dwpe[k], &packed_dwpe[k], seed + idx + k);
    }
    store128(dwpe_tc, packed_dwpe);
}

// ----------------------------------------------------------------------------
// kernel launchers

void encoder_forward(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int N = B * T * C;
    const int grid_size = CEIL_DIV(N, (int)(block_size * x128::size));
    encoder_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

#define BUF_SIZE 5
#define THREAD_BLOCK_NUM_PROC 5

template<typename Type>
__global__ void embedding_backward_kernel2(
    Type *dweight, // VOCAB_SIZE, C
    const Type *dout, // B, T, C
    const int *input_tokens, // B, T
    const int B,
    const int T,
    const int C
) {
    extern __shared__ Type smem[];
    __shared__ uint64_t producer_done_barriers[BUF_SIZE];
    __shared__ uint64_t consumer_done_barriers[BUF_SIZE];

    if (threadIdx.x == 0) {
        for (int i = 0; i < BUF_SIZE; ++i) {
            cuda::ptx::mbarrier_init(&producer_done_barriers[i], 1);

            cuda::ptx::mbarrier_arrive_expect_tx(
                cuda::ptx::sem_release, 
                cuda::ptx::scope_cta, 
                cuda::ptx::space_shared, 
                &producer_done_barriers[i], 
                sizeof(Type) * C
            );

            cuda::ptx::mbarrier_init(&consumer_done_barriers[i], 1);
        }

        cuda::ptx::fence_proxy_async(cuda::ptx::space_cluster);
    } 

    __syncthreads();

    if (threadIdx.x == 0) {
        int n_proc = 0;
        int wait_parity = 1;

        while (n_proc < THREAD_BLOCK_NUM_PROC) {
            for (int idx = 0; idx < BUF_SIZE && n_proc < THREAD_BLOCK_NUM_PROC; ++idx) {
                bool done_waiting = false;
                while (!done_waiting) {
                    done_waiting = cuda::ptx::mbarrier_try_wait_parity(&consumer_done_barriers[idx], wait_parity);
                }

                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_cluster, 
                    cuda::ptx::space_global, 
                    smem + idx * C, 
                    dweight, 
                    C * sizeof(Type), 
                    &producer_done_barriers[idx]
                );

                ++n_proc;
            }

            wait_parity = 1 - wait_parity;
        }
    } else if (threadIdx.x >= WARP_SIZE) {
        int n_proc = 0;
        int wait_parity = 0;

        while (n_proc < THREAD_BLOCK_NUM_PROC) {
            for (int idx = 0; idx < BUF_SIZE && n_proc < THREAD_BLOCK_NUM_PROC; ++idx) {
                bool done_waiting = false;
                while (!done_waiting) {
                    done_waiting = cuda::ptx::mbarrier_try_wait_parity(&producer_done_barriers[idx], wait_parity);
                }

                if (threadIdx.x == WARP_SIZE) {
                    cuda::ptx::cp_reduce_async_bulk(
                        cuda::ptx::space_global, 
                        cuda::ptx::space_shared, 
                        cuda::ptx::op_add,
                        dweight, 
                        smem + idx * C, 
                        C * sizeof(Type)
                    );

                    cuda::ptx::cp_async_bulk_commit_group();
                    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
                    cuda::ptx::mbarrier_arrive(&consumer_done_barriers[idx]);
                }

                ++n_proc;
            }

            wait_parity = 1 - wait_parity;
        }
    }
}

#define MAX_DATA_IDX_SMEM_COPIES 155

template<typename Type>
__global__ void embedding_backward_kernel(
    Type *dweight, // VOCAB_SIZE, DIM
    const Type *dout, // B, T, DIM
    const int *input_tokens, // B, T
    const int (*data)[3],
    const int (*data_thread_block_idxes)[2],
    const int B,
    const int T,
    const int DIM
) {
    extern __shared__ Type smem[]; // X, DIM
    __shared__ uint64_t barrier;
    __shared__ int data_idx_smem_copies[MAX_DATA_IDX_SMEM_COPIES];

    if (threadIdx.x == 0) {
        cuda::ptx::mbarrier_init(&barrier, 1);

        cuda::ptx::fence_proxy_async(cuda::ptx::space_cluster);
    
        int numel_read = 0;

        for (int data_idx = data_thread_block_idxes[blockIdx.x][0]; data_idx <= data_thread_block_idxes[blockIdx.x][1]; ++data_idx) {
            int token    = data[data_idx][0];
            int split    = data[data_idx][1];
            int n_splits = data[data_idx][2];
            int dim = DIM / n_splits;
            int offset = split * dim;

            assert(DIM % n_splits == 0);
            assert(dim % 16 == 0 );

            int smem_copies = 0;

            for (int token_idx = 0; token_idx < B*T; ++token_idx) {
                if (input_tokens[token_idx] == token) {
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_cluster, 
                        cuda::ptx::space_global, 
                        smem + numel_read, 
                        dout + token_idx * DIM + offset, 
                        dim * sizeof(Type), 
                        &barrier
                    );

                    numel_read += dim;
                    ++smem_copies;
                }
            }

            data_idx_smem_copies[data_idx % MAX_DATA_IDX_SMEM_COPIES] = smem_copies;
        }

        cuda::ptx::mbarrier_arrive_expect_tx(
            cuda::ptx::sem_release, 
            cuda::ptx::scope_cta, 
            cuda::ptx::space_shared, 
            &barrier, 
            sizeof(Type) * numel_read
        );
    }

    __syncthreads();

    bool complete = false;
    while (!complete) {
        complete = cuda::ptx::mbarrier_try_wait_parity(&barrier, 0);
    }

    if (threadIdx.x == 0) {
        for (int data_idx = data_thread_block_idxes[blockIdx.x][0]; data_idx <= data_thread_block_idxes[blockIdx.x][1]; ++data_idx) {
            int token    = data[data_idx][0];
            int split    = data[data_idx][1];
            int n_splits = data[data_idx][2];
            int dim = DIM / n_splits;
            int offset = split * dim;
            int smem_token_copies = data_idx_smem_copies[data_idx % MAX_DATA_IDX_SMEM_COPIES];

            for (int i = 0; i < smem_token_copies; ++i) {
                cuda::ptx::cp_reduce_async_bulk(
                    cuda::ptx::space_global, 
                    cuda::ptx::space_shared, 
                    cuda::ptx::op_add, 
                    dweight + token * DIM + offset, 
                    smem + i * dim, 
                    dim * sizeof(Type)
                );
            }
        }
    }
}

bool embedding_backward_initted = false;
int max_dynamic_shared_memory = 230000;

template<typename Type>
void embedding_backward2(
    Type *dweight,
    const Type *dout, 
    const int *input_tokens_device,
    const int *input_tokens_host,
    int B,
    int T,
    int C,
    cudaStream_t stream
) {
    if (!embedding_backward_initted) {
        cudaFuncSetAttribute(embedding_backward_kernel2<floatX>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_shared_memory);
        embedding_backward_initted = true;
    }

    embedding_backward_kernel2<Type><<<114, WARP_SIZE*5, max_dynamic_shared_memory, stream>>>(
        dweight,
        dout, 
        input_tokens_device, 
        B,
        T,
        C
    );
    cudaCheck(cudaGetLastError());
}

std::unordered_map<int, std::pair<int, int>> token_info; // {frequency, current_splits}
int (*embedding_backward_data_host)[3];
int (*embedding_backward_data_device)[3];

int (*embedding_backward_data_thread_block_idxes_host)[2];
int (*embedding_backward_data_thread_block_idxes_device)[2];

template<typename Type>
void embedding_backward(
    Type *dweight,
    const Type *dout, 
    const int *input_tokens_device,
    const int *input_tokens_host,
    int B,
    int T,
    int C,
    cudaStream_t stream
) {
    if (!embedding_backward_initted) {
        cudaFuncSetAttribute(embedding_backward_kernel<floatX>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_shared_memory);

        cudaCheck(cudaMallocHost(&embedding_backward_data_host, 3 * sizeof(int) * B * T));
        cudaCheck(cudaMalloc(&embedding_backward_data_device, 3 * sizeof(int) * B * T));

        cudaCheck(cudaMallocHost(&embedding_backward_data_thread_block_idxes_host, 2 * sizeof(int) * B * T));
        cudaCheck(cudaMalloc(&embedding_backward_data_thread_block_idxes_device, 2 * sizeof(int) * B * T));

        embedding_backward_initted = true;
    }

    // TODO - no guarantee these splits will properly divide DIM and result will be multiple of 16
    int n_data = 0;

    for (int i = 0; i < B * T; ++i) {
        int token = input_tokens_host[i];
        
        auto& [frequency, current_splits] = token_info[token];
        frequency++;

        int required_memory = frequency * C * sizeof(Type);
        int new_splits = (required_memory + max_dynamic_shared_memory - 1) / max_dynamic_shared_memory;

        if (new_splits > current_splits) {
            embedding_backward_data_host[n_data][0] = token;
            embedding_backward_data_host[n_data][1] = new_splits;
            n_data++;
            current_splits = new_splits;
        }
    }

    bool pack = false;
    bool spread = true;
    int n_thread_blocks = 0;

    if (pack) {
        n_thread_blocks = 0;
        int cur_smem = 0;
        embedding_backward_data_thread_block_idxes_host[n_thread_blocks][0] = 0;

        for (int i = 0; i < n_data; ++i) {
            int token = embedding_backward_data_host[i][0];
            auto& [frequency, current_splits] = token_info[token];
            embedding_backward_data_host[i][2] = current_splits;

            int smem_for_data = frequency * C * sizeof(Type) / embedding_backward_data_host[i][2];
            if (cur_smem + smem_for_data > max_dynamic_shared_memory) {
                embedding_backward_data_thread_block_idxes_host[n_thread_blocks][1] = i-1;

                n_thread_blocks++;
                embedding_backward_data_thread_block_idxes_host[n_thread_blocks][0] = i;
                cur_smem = smem_for_data;
            } else {
                cur_smem += smem_for_data;
            }
        }

        embedding_backward_data_thread_block_idxes_host[n_thread_blocks][1] = n_data-1;
    } else if (spread) {
        int target_memory_per_thread_block = B * T * C * sizeof(Type) / 114;

        if (target_memory_per_thread_block > max_dynamic_shared_memory) {
            target_memory_per_thread_block = max_dynamic_shared_memory;
        }

        n_thread_blocks = 0;
        int cur_smem = 0;
        embedding_backward_data_thread_block_idxes_host[n_thread_blocks][0] = 0;

        for (int i = 0; i < n_data; ++i) {
            int token = embedding_backward_data_host[i][0];
            auto& [frequency, current_splits] = token_info[token];
            embedding_backward_data_host[i][2] = current_splits;

            int smem_for_data = frequency * C * sizeof(Type) / embedding_backward_data_host[i][2];

            if (cur_smem + smem_for_data > target_memory_per_thread_block) {
                int end_idx = max(embedding_backward_data_thread_block_idxes_host[n_thread_blocks][0], i-1);
                int next_start_idx = min(end_idx+1, n_data-1);
                embedding_backward_data_thread_block_idxes_host[n_thread_blocks][1] = end_idx;
                embedding_backward_data_thread_block_idxes_host[n_thread_blocks+1][0] = next_start_idx;

                n_thread_blocks++;
                cur_smem = smem_for_data;
            } else {
                cur_smem += smem_for_data;
            }
        }

        embedding_backward_data_thread_block_idxes_host[n_thread_blocks][1] = n_data-1;
    } else {
        n_thread_blocks = n_data;
        for (int i = 0; i < n_data; ++i) {
            int token = embedding_backward_data_host[i][0];
            auto& [frequency, current_splits] = token_info[token];
            embedding_backward_data_host[i][2] = current_splits;

            embedding_backward_data_host[i][0] = i;
            embedding_backward_data_host[i][1] = i;
        }
    }

    for (int i = 0; i < n_thread_blocks; ++i) {
        assert(embedding_backward_data_thread_block_idxes_host[i][1] - embedding_backward_data_thread_block_idxes_host[i][0] < MAX_DATA_IDX_SMEM_COPIES);
    }

    // for (int i = 0; i < n_data; ++i) {
    //     printf("embedding_backward_data_host[%d] = {%d, %d, %d}\n", i, embedding_backward_data_host[i][0], embedding_backward_data_host[i][1], embedding_backward_data_host[i][2]);
    // }

    // printf("*****************\n");

    // printf("n_thread_blocks = %d\n", n_thread_blocks);

    // for (int i = 0; i <= n_thread_blocks; ++i) {
    //     printf("embedding_backward_data_thread_block_idxes_host[%d] = {%d, %d}\n", i, embedding_backward_data_thread_block_idxes_host[i][0], embedding_backward_data_thread_block_idxes_host[i][1]);
    // }

    // printf("*****************\n");

    // exit(1);

    cudaCheck(cudaMemcpyAsync(embedding_backward_data_device, embedding_backward_data_host, 3 * sizeof(int) * n_data, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(embedding_backward_data_thread_block_idxes_device, embedding_backward_data_thread_block_idxes_host, 2 * sizeof(int) * n_thread_blocks, cudaMemcpyHostToDevice, stream));

    embedding_backward_kernel<Type><<<n_thread_blocks, 128, max_dynamic_shared_memory, stream>>>(
        dweight,
        dout, 
        input_tokens_device, 
        embedding_backward_data_device,
        embedding_backward_data_thread_block_idxes_device,
        B,
        T,
        C
    );
    cudaCheck(cudaGetLastError());

    token_info.clear();
}

void encoder_backward_tma(floatX* dwte, floatX* dwpe, floatX* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const floatX* dout, const int* inp, const int* inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed, cudaStream_t stream) {
    NVTX_RANGE_FN();

    const int block_size = 256;
    const int N = T * C / x128::size;
    const int grid_size = CEIL_DIV(N, block_size);
    wpe_backward_kernel<<<grid_size, block_size, 0, stream>>>(dwpe, dout, inp, B, T, C, seed);
    cudaCheck(cudaGetLastError());

    embedding_backward2(dwte, dout, inp, inputs_cpu, B, T, C, stream);
}

// Fully deterministic (see comments in wte_backward_kernel and wpe_backward_kernel for more details)
void encoder_backward(floatX* dwte, floatX* dwpe, floatX* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const floatX* dout, const int* inp, const int* inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed, cudaStream_t stream) {
    NVTX_RANGE_FN();

    // Launch wpe kernel first (so it runs on the GPU in parallel with the CPU pre-processing for wte)
    const int block_size = 256;
    const int N = T * C / x128::size;
    const int grid_size = CEIL_DIV(N, block_size);
    wpe_backward_kernel<<<grid_size, block_size, 0, stream>>>(dwpe, dout, inp, B, T, C, seed);
    cudaCheck(cudaGetLastError());

    // check the GPU scratch buffer is large enough to hold the bucket info and workload indices
    // todo - this is trivially true given hardcoded scratch buffer size here, is this useful?
    int num_c_groups = CEIL_DIV(C, x128::size * WARP_SIZE);
    assert(B*T*num_c_groups * (sizeof(int4)+sizeof(int)) <= B*T*3*C * sizeof(floatX));

    // Step 1: Sort inputs into buckets
    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    for (uint64_t bt = 0; bt < B * T; bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            // todo - passing c_group/inputs_cpu[bt] in data to avoid a second hash lookup is a bit hacky
            uint64_t data = bt + (c_group<<32ULL) + ((uint64_t)inputs_cpu[bt]<<42ULL);
            buckets[c_group + num_c_groups * inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }

    // Step 2: Sort buckets by size in descending order
    // this is so the largest buckets are processed first by the GPU
    // otherwise, if they started late, they would still be running with the rest of the GPU idle
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(), // ugly because we don't have a typedef for the std::pair
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a, const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    int num_buckets = buckets.size();
    int bucket_index = 0;
    int workload_index = 0;
    for (const auto& bucket : sortedBuckets) {
        bucket_info[bucket_index].x = workload_index; // bucket start
        bucket_info[bucket_index].y = bucket.second.size(); // bucket size
        bucket_info[bucket_index].z = (bucket.second[0] >> 42ULL) & ((1ULL<<20ULL)-1); // bucket ix
        bucket_info[bucket_index].w = (bucket.second[0] >> 32ULL) & ((1ULL<<10ULL)-1); // bucket c

        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL<<31ULL)-1ULL));
        }
        bucket_index++;
    }

    // Step 3: Copy data from host to device (async until the last one to avoid synchronising CPU/GPU twice)
    // todo - could use CUDA events (even without streams) to avoid CPU/GPU synchronisation completely
    int4* d_bucket_info = (int4*)scratch;
    int*  d_workload_indices = (int*)(scratch + B*T*num_c_groups * sizeof(int4));
    cudaCheck(cudaMemcpyAsync(d_bucket_info, bucket_info, num_buckets * sizeof(int4), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Launch wte kernel
    // todo - profile block sizes on more content (depends on number of buckets and on GPU?)
    wte_backward_kernel<256><<<num_buckets, 256, 0, stream>>>(dwte, d_bucket_info, d_workload_indices, dout, inp, seed, B, T, C);
    cudaCheck(cudaGetLastError());
}
