#include <stdio.h>
#include <stdint.h>
//#include <chrono>

#include "generator.h"

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__device__ uint64_t _mcStepSeed(uint64_t s, uint64_t salt)
{
    return s * (s * 6364136223846793005ULL + 1442695040888963407ULL) + salt;
}

__device__ int _mcFirstIsZero(uint64_t s)
{
    return (int)(((int64_t)s >> 24) % 10) == 0;
}

__device__ uint64_t _getChunkSeed(uint64_t ss, int x, int z)
{
    uint64_t cs = ss + x;
    cs = _mcStepSeed(cs, z);
    cs = _mcStepSeed(cs, x);
    cs = _mcStepSeed(cs, z);
    return cs;
}

__device__ uint64_t _getStartSalt(uint64_t ws, uint64_t ls)
{
    uint64_t st = ws;
    st = _mcStepSeed(st, ls);
    st = _mcStepSeed(st, ls);
    st = _mcStepSeed(st, ls);
    return st;
}

__device__ uint64_t _getStartSeed(uint64_t ws, uint64_t ls)
{
    uint64_t ss = ws;
    ss = _getStartSalt(ss, ls);
    ss = _mcStepSeed(ss, 0);
    return ss;
}

__device__ uint64_t _getLayerSalt(uint64_t salt)
{
    uint64_t ls = _mcStepSeed(salt, salt);
    ls = _mcStepSeed(ls, salt);
    ls = _mcStepSeed(ls, salt);
    return ls;
}


__device__ int isValid(uint64_t ss, int radius)
{
    //sung to ickle me pickle me tickle me too

    //this is radius
    uint64_t cs;
    for(int x = -radius; x < radius; x++)
    {
        //calculate how tall this slice is
        int dz = (int)sqrtf(radius*radius - x*x);
        for(int z = -dz; z < dz; z++)
        {
            cs = _getChunkSeed(ss, x, z);

            //maybe have all these run to get good warps
            if(_mcFirstIsZero(cs) != 0) return 0;
        }
    }    

    return 1;
}

//this tracks how many seeds have been found across this kernel invocation
__device__ uint64_t d_deviceSeedCount;
__device__ uint64_t d_seedArray[1024];

//yes the 1024 thing is horrible i dont care
__global__ void spawnThreads(uint64_t block, int radius)
{
    //d_deviceSeeds starts empty, and gets filled up. d_deviceSeeds
    //is shared across all threads in a block, and it will be returned
    //along with the count of seeds found
    if(threadIdx.x == 0) d_deviceSeedCount = 0;
    __syncthreads(); //sync threads in this block

    //this should search 2^32 seeds. that will be one block
    const uint64_t startSeed = (uint64_t)block << 32;
    const uint64_t endSeed = startSeed + (1ULL << 32);
    const uint64_t layerSalt = _getLayerSalt(1);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t ss;
    for(uint64_t ws = startSeed + (uint64_t)tid; ws < endSeed; ws += blockDim.x*gridDim.x)
    {
        //if we forego this line, we can compute worldSeeds from valid startSeeds
        //(its not really faster)
        ss = _getStartSeed(ws, layerSalt);
        if(isValid(ss, radius))
        {
            //use d_deviceSeedCount to add ws to last spot
            d_seedArray[d_deviceSeedCount] = ws;

            //possibly increment an atomic counter and return that
            //rage inducing
            atomicAdd((unsigned long long*)&d_deviceSeedCount, 1);
        }
    }
}

void printProgress(uint64_t startBlock, uint64_t currentBlock, uint64_t seedsFound)
{
    currentBlock -= startBlock;
    //block | current seeds found(currently zero) | seedrate
    fprintf(stderr, "\rblock %lu | %lu seeds searched | %lu seeds found",
        currentBlock + startBlock, currentBlock << 32, seedsFound);
}

int b18_hasNoCenterIsland(uint64_t seed)
{
    //uses cubiomes
    Generator g;
    setupGenerator(&g, MC_B1_8, 0);
    applySeed(&g, DIM_OVERWORLD, seed);
    const int scale = 256;
    const int bound = 8192 / scale;
    const Range range = {scale, -bound, -bound, 2*bound, 2*bound, 0, 0};
    int* cache = allocCache(&g, range);
    genBiomes(&g, cache, range);

    //now we look through cache to see if its all ocean. and...
    //ocean is saved in cubiomes as an enum.
    // ocean = 0
    //printf("%d\n", cacheSize);
    for(int i = 0; i < 4*bound*bound; i++)
    {
        if(cache[i] != ocean)
        {
            free(cache);
            return 0;
        }
    }
    free(cache);
    return 1;
}

int mc10_hasNoCenterIsland(uint64_t seed)
{
    //uses cubiomes
    Generator g;
    setupGenerator(&g, MC_1_0_0, 0);
    applySeed(&g, DIM_OVERWORLD, seed);
    const int scale = 256;
    const int bound = 8192 / scale;
    const Range range = {scale, -bound, -bound, 2*bound, 2*bound, 0, 0};
    int* cache = allocCache(&g, range);
    genBiomes(&g, cache, range);

    //now we look through cache to see if its all ocean. and...
    //ocean is saved in cubiomes as an enum.
    // ocean = 0
    //printf("%d\n", cacheSize);
    for(int i = 0; i < 4*bound*bound; i++)
    {
        if(cache[i] != ocean && cache[i] != mushroom_fields)
        {
            free(cache);
            return 0;
        }
    }
    free(cache);
    return 1;
}

int main(int argc, char** argv)
{
    //printf("%d\n", b18_hasNoCenterIsland(5828078988));

    //return 0;

    if(argc != 6)
    {
        printf("%s <startSeed> <endSeed> <radius in 1:8192 tiles> <gpu_id> <filename>\n", argv[0]);
        return 1;
    }

    uint64_t startSeed = atoll(argv[1]);
    uint64_t endSeed = atoll(argv[2]);
    int radius = atoi(argv[3]);
    int device_id = atoi(argv[4]);

    uint64_t startBlock = startSeed >> 32;
    uint64_t endBlock = endSeed >> 32;

    // Optimized launch configuration
    int blockSize;
    int numBlocks;
    
    FILE* fp = fopen(argv[5], "a");

    cudaSetDevice(device_id);

    // Get device properties for optimal configuration
    //cudaDeviceProp prop;
    //CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    // Calculate optimal number of blocks based on SM count
    int minGridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                                   spawnThreads, 0, 0));
    numBlocks = minGridSize;

    //fprintf(stderr, "Processing seeds with %d blocks of %d threads...\n", numBlocks, blockSize);
    //fprintf(stderr, "GPU: %s with %d SMs\n", prop.name, prop.multiProcessorCount);
    
    //using namespace std::chrono;
    //auto start = high_resolution_clock::now();

    //i have no idea what type stop is
    //auto stop = high_resolution_clock::now();

    uint64_t resultCount = 0;
    uint64_t h_seedCount;

    //if you get segfaults increase this.
    //this is only going to happen if you
    //manage to get extremely lucky
    //and get 1024 seeds in 2^32
    uint64_t h_seedArray[1024];

    for(uint64_t block = startBlock; block < endBlock; block++)
    {
        spawnThreads<<<numBlocks, blockSize>>>(block, radius);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
            return -1;
        }

        //copy seed count and actual returned seeds to host
        cudaMemcpyFromSymbol(&h_seedCount, d_deviceSeedCount, sizeof(uint64_t));
        cudaMemcpyFromSymbol(&h_seedArray, d_seedArray, 1024 * sizeof(uint64_t));

        //loop through seeds and print them
        //we need to use cubiomes here, not looking forward to it
        for(int i = 0; i < h_seedCount; i++)
        {
            if(b18_hasNoCenterIsland(h_seedArray[i]) || mc10_hasNoCenterIsland(h_seedArray[i]))
            {
                fprintf(fp, "%ld\n", h_seedArray[i]);
                fflush(fp);
                resultCount++;
            }
            
        }

        //sync
        CUDA_CHECK(cudaDeviceSynchronize());
        //stop = high_resolution_clock::now();
        printProgress(startBlock, block, resultCount);
    }
    //fprintf(stderr, "\n");

    //auto duration = duration_cast<milliseconds>(stop - start);
    
    //fprintf(stderr, "Total time (CPU+GPU): %ld ms\n", duration.count());

    fclose(fp);

    return 0;
}
