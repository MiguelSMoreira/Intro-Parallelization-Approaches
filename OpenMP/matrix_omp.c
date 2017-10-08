#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*****************************************************
the following function generates a "size"-element vector
and a "size x size" matrix
 ****************************************************/
void matrix_gen(int size, double *matrix){
  int i;
  for(i=0; i<size*size; i++)
      matrix[i] = ((double)rand())/5307.0;
}

/****************************************************

 ***************************************************/
void matrix_mult_sq(int size, double *matrix1_in,
		       double *matrix2_in, double *matrix_out){
  int rows, cols;
  int j;

  // For cycle added to implement a sequential, matrix by matrix, multiplication that repeats the code given for the vector by matrix multipication for every row of the first operand's matrix
  for(rows=0; rows<size; rows++){
    for(cols=0; cols<size; cols++){
      matrix_out[rows*size + cols] = 0.0;
      for(j=0; j<size; j++)
        matrix_out[rows*size + cols] += matrix1_in[ rows*size + j ] * matrix2_in[j*size+ cols];
    }
  }
}

/****************************************************

 ***************************************************/
void matrix_mult_pl(int size, double *matrix1_in,
		       double *matrix2_in, double *matrix_out){
  int row, col;
  int j, i;
    
// Changes the for loop behavior to a more (the most) parallelizable implementation, where each entry of the resulting matrix is calculated by a single thread.
# pragma omp parallel				\
    shared(size, matrix1_in, matrix2_in, matrix_out, i)	\
    private(row, col, j)
# pragma omp for
    // Each thread is assigned a row of the first operand and a column of the second operand, that is computes (multiplying its entries and adding them together) to achieve the final result for that specific entry in the resulting matrix
    for(i=0; i<size*size; i++){
        row= (i/size);
        col= (i%size);
        matrix_out[i] = 0.0;
        for(j=0; j<size; j++){
            matrix_out[i] += matrix1_in[ (row*size)+j ] * matrix2_in[ col+ (j*size) ];
        }
    }
}



/****************************************************
 main
 ***************************************************/
int main(int argc, char *argv[]){
  if(argc < 2){
    printf("Usage: %s matrix/vector_size\n", argv[0]);
    return 0;
  }

  int m, n;
  int size = atoi(argv[1]);
    
  // Allocates two vectors of size*size dimention, that will serve as the matrix data types for the computation. Same for the vectors that will hold the result.
  double *matrix1 = (double *)malloc(sizeof(double)*size*size);
  double *matrix2 = (double *)malloc(sizeof(double)*size*size);
    
  double *result_sq = (double *)malloc(sizeof(double)*size*size);
  double *result_pl = (double *)malloc(sizeof(double)*size*size);
    
  matrix_gen(size, matrix1);
  matrix_gen(size, matrix2);
    
  double time_sq = 0;
  double time_pl = 0;
    
  time_sq = omp_get_wtime();
  matrix_mult_sq(size, matrix1, matrix2, result_sq);
  time_sq = omp_get_wtime() - time_sq;

  time_pl = omp_get_wtime();
  matrix_mult_pl(size, matrix1, matrix2, result_pl);
  time_pl = omp_get_wtime() - time_pl;

  printf("SEQUENTIAL EXECUTION: %f (sec)\n", time_sq);
  printf("PARALLEL EXECUTION WITH %d (threads) ON %d (processors): %f (sec)\n",
	 omp_get_max_threads(), omp_get_num_procs(), time_pl);

  //check
  int i;
  for(i=0; i<size; i++)
    if(result_sq[i] != result_pl[i]){
      printf("wrong at position %d\n", i);
      return 0;
    }

  free(matrix1);
  free(matrix2);
  free(result_sq);
  free(result_pl);
  return 1;
}
