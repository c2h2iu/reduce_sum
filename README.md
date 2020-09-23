# reduce_sum

Cuda program for reduce sum of an array. The codes are based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

# Usage:      
./reduce approach elements     

"approach" is from 0 to 5. 0 represents sequential execution. 1 to 5 corresponds to the approaches described in the above link.

"elements" represent the array size which is the power of 2 only.
