// gcc -shared -fPIC -o model.so model.c
// nvcc -shared -Xcompiler -fPIC model.cu -o model.so

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// params_7b = ModelArgs(
//     dim=4096, 
//     n_layers=32, 
//     n_heads=32, 
//     n_kv_heads=None, 
//     vocab_size=65536, 
//     ffn_dim_multiplier=1.0,
//     multiple_of=256, 
//     norm_eps=1e-05, 
//     rope_theta=10000.0, 
//     qk_normalization=True, 
//     swin_norm=False
// )

#define DIM 4096
#define VOCAB_SIZE 65536

#define B 2
#define T 256
#define C DIM

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

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

template<class floatX>
__global__ void embedding_forward_kernel(floatX* out, const uint16_t* inp, const floatX* wte) {
    typedef Packed128<floatX> x128;

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

    x128 packed_out;
    x128 wte128 = load128cs(wte_ix);
    for (int k = 0; k < x128::size; k++) {
        packed_out[k] = (floatX)((float)wte128[k]);
    }
    store128(out_btc, packed_out);
}


template<class floatX>
struct Embedding {
  floatX *weight;

  uint16_t *inp;
  floatX *out;

  Embedding(floatX *_weight) {
    cudaCheck(cudaMalloc((void**)&weight, VOCAB_SIZE * DIM * sizeof(floatX)));

    cudaCheck(cudaMalloc((void**)&inp, B * T * sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void**)&out, B * T * C * sizeof(floatX)));

    // TMP
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        initArray(weight + i * DIM, DIM, static_cast<floatX>(i));
    }
  }

  void forward() {
        typedef Packed128<floatX> x128;
        const int block_size = 256;
        const int N = B * T * C;
        const int grid_size = CEIL_DIV(N, (int)(block_size * x128::size));
        embedding_forward_kernel<<<grid_size, block_size, 0>>>(out, inp, weight);
        cudaCheck(cudaGetLastError());
  }
};

template<class floatX>
struct Model {
    Embedding<floatX> embedding;
    Model() = default;

    float* forward(uint16_t inp_host[B * T]) {
        for (size_t i = 0; i < (B * T); i++) {
            printf("%d ", inp_host[i]);
        }
        printf("\n");

        cudaCheck(cudaMemcpy(embedding.inp, inp_host, B * T * sizeof(uint16_t), cudaMemcpyHostToDevice));

        embedding.forward();

        floatX *dbg = (floatX*)malloc(B * T * C * sizeof(floatX));
        cudaCheck(cudaMemcpy(dbg, embedding.out, B * T * C * sizeof(floatX), cudaMemcpyDeviceToHost));

        return dbg;
    }
};

Model<float> _model;

extern "C" float* model(uint16_t numbers[B*T]) {
    return _model.forward(numbers);
}