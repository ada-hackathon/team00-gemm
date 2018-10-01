__kernel  __attribute__ ((reqd_work_group_size(8,8,1)))
void gemm(__global const float* A,__global const float* B,__global  float* C,const int size){
             //        int i = get_global_id(0);
//             //         int j = get_global_id(1);
//             //         int k ;
//             //         float temp = 0;
////                      int tile = get_local_size(0);
                      const int tile = 8;
////                      extern __shared__ cuDoubleComplex temp[];
                      __local float temp_A[tile*tile];
                      __local float temp_B[tile*tile];
//      //                int tile = blockDim.x;
//                      printf("global id = %d, local id = %d")
                      int row = get_global_id(1);//*tile+get_local_id(1);
                      int col = get_global_id(0);//*tile+get_local_id(0);
                      int numtile = size/tile;
////                      if(size%tile)numtile++;
                      float reduce = 0;
                      int index = get_local_id(1)*tile+get_local_id(0);
                      int block = tile*tile;
                      int start = 0;
                      for(int k = 0; k < numtile; k++){
////                        if(row < size && start+get_local_id(0) < size)
                            temp_A[index] = A[row*size+start+get_local_id(0)];
////                        if(col < size && start+get_local_id(1) < size)
                          temp_B[index] = B[(start+get_local_id(1))*size+col];
                        barrier (CLK_LOCAL_MEM_FENCE);
////                        if(col < L && row < M)
                          for(int i = 0; i < tile /*&& start + i < size*/; i++)
                        	reduce += temp_A[get_local_id(1)*tile+i] * temp_B[i*tile+get_local_id(0)];
////                            reduce = cuCadd(reduce,cuCmul(temp[get_local_id(1)*tile+i],temp[block+i*tile+get_local_id(0)]));
						barrier (CLK_LOCAL_MEM_FENCE);
                        start = start+tile;
                      }
////                      if(col < L && row < M)
                    	  C[row*size+col] = reduce;

}
