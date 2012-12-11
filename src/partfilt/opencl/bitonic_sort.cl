/*
This file contains code derived from the NVIDIA CUDA SDK. From its EULA, with respect to Source Code:

Source Code:  Developer shall have the right to modify and create derivative works with the Source Code.  Developer shall own any derivative works ("Derivatives") it creates to the Source Code, provided that Developer uses the Materials in accordance with the terms and conditions of this Agreement.  Developer may distribute the Derivatives, provided that all NVIDIA copyright notices and trademarks are propagated and used properly and the Derivatives include the following statement: "This software contains source code provided by NVIDIA Corporation."
*/

// Based on
// Bitonic sort
// http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm

inline void comparator(__local float* weights, int posA, int posB, __local int* const index, const unsigned int dir)
{
	float tmpWeight;
	int tmpPos;

	if((weights[posA] > weights[posB]) == dir)
        {
		tmpWeight = weights[posA];
		weights[posA] = weights[posB];
		weights[posB] = tmpWeight;

		tmpPos = index[posA];
		index[posA] = index[posB];
		index[posB] = tmpPos;
        }
}

inline void bitonic_sort(
	__local float* const d_weights,
	__local int* const index,
	const int num_values,
	const unsigned int dir)
{

	const unsigned int tid = get_local_id(0);
	barrier(CLK_LOCAL_MEM_FENCE);

	for(unsigned int size = 2; size < num_values; size <<= 1)
	{
		//Bitonic merge
		unsigned int ddd = dir ^ ( (tid & (size / 2)) != 0 );
		for(unsigned int stride = size / 2; stride > 0; stride >>= 1)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			unsigned int pos = 2 * tid - (tid & (stride - 1));
			comparator(d_weights, pos+0, pos+stride, index, ddd);
		}
	}

	//ddd == dir for the last bitonic merge step
	{
		for(unsigned int stride = num_values / 2; stride > 0; stride >>= 1)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			unsigned int pos = 2 * tid - (tid & (stride - 1));
			comparator(d_weights, pos+0, pos+stride, index, dir);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
}

