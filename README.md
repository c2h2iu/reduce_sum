# reduce_five_approaches

Cuda program for reduce sum of an array. The codes are based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf


# Usage:      
./reduce_five_approaches approach elements     

"approach" is from 0 to 5. 0 represents sequential execution. 1 to 5 corresponds to the approaches described in the above link.

"elements" represent the array size which is the power of 2 only.

----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------

# reduce_api
This api is develped based on the approach 5 introduced in the above document. An example code demonstrates the usage. This api allows arbitrary array length from 1 to 1800000.

# Usage:
./example length

"length" represents the array size which is an arbitrary integer number from 1 to 1,800,000 at least. 
