// nvcc -shared -Xcompiler -fPIC model2.cu -o model.so

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using namespace std;

#define DIM 4096
#define VOCAB_SIZE 65536

#define B 2
#define T 256
#define C DIM

#define WARP_SIZE 32U

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// Packed128 data structure that forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__  static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }
    __device__ static Packed128 zeros() {
        return constant(0.f);
    }
    __device__ static Packed128 ones() {
        return constant(1.f);
    }

    __device__ ElementType& operator[](int index) {
        return payload[index];
    }
    __device__ const ElementType& operator[](int index) const {
        return payload[index];
    }
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};

typedef Packed128<__nv_bfloat16> x128;

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

template<class Type>
__global__ void initArrayKernel(Type *arr, int size, Type value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = value;
    }
}

template<class Type>
void initArray(Type *arr, int size, Type value) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(size, block_size);
    initArrayKernel<<<grid_size, block_size>>>(arr, size, value);
    cudaCheck(cudaGetLastError());
}

__global__ void toBfloat16Kernel(__nv_bfloat16 *to, float *from, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        to[idx] = from[idx];
    }
}

void toBfloat16(__nv_bfloat16 *to, float *from, int size) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(size, block_size);
    toBfloat16Kernel<<<grid_size, block_size>>>(to, from, size);
    cudaCheck(cudaGetLastError());
}

__global__ void bfloat16ToFloat32Kernel(float *to, __nv_bfloat16 *from, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        to[idx] = __bfloat162float(from[idx]);
    }
}

void bfloat16ToFloat32(float *to, __nv_bfloat16 *from, int size) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(size, block_size);
    bfloat16ToFloat32Kernel<<<grid_size, block_size>>>(to, from, size);
    cudaCheck(cudaGetLastError());
}

template<class Type>
__global__ void embedding_forward_kernel(const Type* weight, const uint16_t *inp, Type* out) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;
    if (idx >= N) { return; }

    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;

    int ix = inp[b * T + t];

    Type* out_btc = out + b * T * C + t * C + c;
    const Type* weight_ix = weight + ix * C + c;

    x128 packed_out;
    x128 weight128 = load128cs(weight_ix);
    for (int k = 0; k < x128::size; k++) {
        packed_out[k] = (Type)((float)weight128[k]);
    }
    store128(out_btc, packed_out);
}

template<class Type>
void embedding_forward(const Type *weight, const uint16_t *inp, Type *out) {
    const int block_size = 256;
    const int N = B * T * C;
    const int grid_size = CEIL_DIV(N, (int)(block_size * x128::size));
    embedding_forward_kernel<<<grid_size, block_size, 0>>>(weight, inp, out);
    cudaCheck(cudaGetLastError());
}

template <int BLOCK_SIZE=256>
__global__ void embedding_backward_kernel(floatX* dwte,
                                    const int4* bucket_info, const int* workload_indices, const floatX* dout) {
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

template<class floatX>
void embedding_backward(floatX* dwte, floatX* dwpe, floatX* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const floatX* dout, const int* inputs_cpu, // cpu/gpu inputs
                      ) {
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
    cudaCheck(cudaMemcpyAsync(d_bucket_info, bucket_info, num_buckets * sizeof(int4), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int), cudaMemcpyHostToDevice));

    // Launch wte kernel
    // todo - profile block sizes on more content (depends on number of buckets and on GPU?)
    wte_backward_kernel<256><<<num_buckets, 256, 0>>>(dwte, d_bucket_info, d_workload_indices, dout);
    cudaCheck(cudaGetLastError());
}

#pragma pack(push, 1)
template<class Type>
struct _Weights {
    Type embedding[VOCAB_SIZE * DIM];
    Type norm[DIM];
    Type output[DIM * VOCAB_SIZE];
};

typedef _Weights<float> WeightsFull;
typedef _Weights<__nv_bfloat16> WeightsHalf;
size_t weights_numel = sizeof(WeightsFull) / sizeof(float);

struct Optim {
    WeightsFull first_moment;
    WeightsFull second_moment;
};
size_t optim_numel = sizeof(Optim) / sizeof(float);
#pragma pack(pop)
    
template<class Type>
struct NetState {
    Type embedding[B * T * C];
};

typedef NetState<float> NetStateFull;
typedef NetState<__nv_bfloat16> NetStateHalf;
size_t net_state_numel = sizeof(NetStateFull) / sizeof(float);

WeightsFull *weights_full;
WeightsHalf *weights_half;
WeightsHalf *grads;
Optim *optim;

uint16_t *input;
NetStateHalf *net_state;

extern "C" void init() {
    cout << sizeof(WeightsFull) << endl;
    cudaCheck(cudaMalloc((void**)&weights_full, sizeof(WeightsFull)));

    int fd = open("ckpt.bin", O_RDONLY);
    WeightsFull *checkpoint = static_cast<WeightsFull*>(mmap(nullptr, sizeof(WeightsFull), PROT_READ, MAP_PRIVATE, fd, 0));
    cudaCheck(cudaMemcpy(weights_full, checkpoint, sizeof(WeightsFull), cudaMemcpyHostToDevice));
    munmap(checkpoint, sizeof(WeightsFull));
    close(fd);

    cudaCheck(cudaMalloc((void**)&weights_half, sizeof(WeightsHalf)));
    toBfloat16((__nv_bfloat16*)weights_half, (float*)weights_full, weights_numel);

    cudaCheck(cudaMalloc((void**)&grads, sizeof(WeightsHalf)));
    cudaCheck(cudaMemset(grads, 0, weights_numel));

    cudaCheck(cudaMalloc((void**)&optim, sizeof(Optim)));
    cudaCheck(cudaMemset(optim, 0, optim_numel));

    cudaCheck(cudaMalloc((void**)&input, B*T*sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void**)&net_state, sizeof(NetStateHalf)));
}

extern "C" float* model(uint16_t inp_host[B*T]) {
    for (size_t i = 0; i < (B * T); i++) {
        printf("%d ", inp_host[i]);
    }
    printf("\n");

    cudaCheck(cudaMemcpy(input, inp_host, B*T*sizeof(uint16_t), cudaMemcpyHostToDevice));
    embedding_forward(weights_half->embedding, input, net_state->embedding);

    // return 32.0f;

    // WeightsFull *host_weights = (WeightsFull*)malloc(sizeof(WeightsFull));
    // cudaCheck(cudaMemcpy(host_weights, weights_full, sizeof(WeightsFull), cudaMemcpyDeviceToHost));
    // return host_weights->embedding;

    NetStateFull *net_state_full;
    cudaCheck(cudaMalloc((void**)&net_state_full, sizeof(NetStateFull)));
    bfloat16ToFloat32((float*)net_state_full, (__nv_bfloat16*)net_state, net_state_numel);

    NetStateFull *net_state_host = (NetStateFull*)malloc(sizeof(NetStateFull));
    cudaCheck(cudaMemcpy(net_state_host, net_state_full, sizeof(NetStateFull), cudaMemcpyDeviceToHost));
    return net_state_host->embedding;
}
