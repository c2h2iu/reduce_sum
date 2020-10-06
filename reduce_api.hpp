#include <iostream>
#include <vector>
#include <random>
#include <algorithm>


namespace chiu{
    int numThreads = 256;
    unsigned int numBlocks = 0;
    

    template<typename T, typename C>
    __device__ void warpReduce(volatile T* sdata, const T& tid, const T& length, C&& bop){
        if((length > 32) && (tid + 32 < length))    sdata[tid] = bop(sdata[tid], sdata[tid + 32]);
        if((length > 16) && (tid + 16 < length))    sdata[tid] = bop(sdata[tid], sdata[tid + 16]);
        if((length > 8)  && (tid + 8  < length))    sdata[tid] = bop(sdata[tid], sdata[tid + 8]);
        if((length > 4)  && (tid + 4  < length))    sdata[tid] = bop(sdata[tid], sdata[tid + 4]);
        if((length > 2)  && (tid + 2  < length))    sdata[tid] = bop(sdata[tid], sdata[tid + 2]);
        if((length > 1)  && (tid + 1  < length))    sdata[tid] = bop(sdata[tid], sdata[tid + 1]);
    }


    template<typename T, typename C>
    __global__ void kernel(T* d_in, T* d_out, const unsigned int length, C&& bop){
        extern __shared__ T sdata[];
  
        T tid = threadIdx.x;
        T i   = threadIdx.x + blockIdx.x * (blockDim.x * 2);

        sdata[tid] = 0;
       
        if(i < length){
            if(gridDim.x > 1){
                if(i + blockDim.x < length)    sdata[tid] = bop(d_in[i], d_in[i + blockDim.x]);
                else                           sdata[tid] = d_in[i];
            }
            else    sdata[tid] = d_in[i];
        }
        __syncthreads();


       for(T s = blockDim.x/2; s > 32; s>>= 1){
           if(tid < s)    sdata[tid] = bop(sdata[tid], sdata[tid + s]);
           __syncthreads();
       }

       if(tid < 32)    warpReduce(sdata, tid, length, bop);

       if(tid == 0)    d_out[blockIdx.x] = sdata[0];
    }

    
    template<typename I, typename T, typename C>      
    T reduce(I beg, I end, T& init, C&& bop){
        unsigned int length = std::distance(beg, end); 
        
        if(length == 0)    return init; 
        if(length == 1)    return bop(init, *beg);

        using U = typename std::decay<decltype(*beg)>::type; 
  
        U size = length * sizeof(U);
        U out_size = ((length-1)/numThreads+1) * sizeof(U); 

        std::vector<U> h_out(out_size, 0);
  
        U* d_in;
        U* d_out;

        cudaMalloc(&d_in,  size);
        cudaMalloc(&d_out, out_size);

        cudaMemcpy(d_in, &(*beg), size, cudaMemcpyHostToDevice);
         
        while(length > 1){
            if(length < numThreads)    numBlocks = 1;
            else                       numBlocks = (length - 1) / numThreads + 1;
  
            kernel<<<numBlocks, numThreads, numThreads * sizeof(U)>>>(d_in, d_out, length, bop);
      
            length = (numBlocks - 1) / 2 + 1;

            d_in = d_out;
        }

        cudaMemcpy(h_out.data(), d_out, out_size, cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);

        return bop(init, h_out[0]);
    }
}



template<typename T, typename I, typename C>
T verify(const std::vector<T>& array, I& init, C&& bop){
    T res = 0;
    for(auto a : array)    res = bop(res, a);
 
    return bop(res, init);
}



template<typename T>
void gen_array(std::vector<T>& array){
    std::default_random_engine defEngine(time(0));
    std::uniform_int_distribution<int> intDistro(0, 100);
    std::for_each(array.begin(), array.end(), [&](T& n){ 
        n = intDistro(defEngine); 
    });
}

