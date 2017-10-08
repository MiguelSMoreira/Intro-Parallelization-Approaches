#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <xmmintrin.h>
#include <time.h>
#include <omp.h>

/*****************************************************
the following function generates a "size"-element vector
and a "size x size" matrix
 ****************************************************/
void matrix_gen(int size, int adsize, double *matrix){
  int i;
    
  // Loop that fills the matrix structure.
  // Verifies if the matrix size is a multiple of 2. If it is not, it will fill the padding columns and columns with zeros, in order not to influence the calculation results
  for(i=0; i<adsize*adsize; i++){
    // size!=adsize verifies if entry is padding
    // At most 1 columns will have to be filled with zeros, in accordance with the remainder of the division between adsize and two.
    // All rows which satisfy i>size*adsize-1 are serving as pading (filled w/zeros)
    if( size!=adsize && (((i+1)%adsize==0 &&(adsize-size >0)) || i>(size*adsize-1)) ) matrix[i] = 0;
    else matrix[i] = i*1.3f + 1;//((float)rand())/5307.0f;
  }
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
        matrix_out[rows*size + cols] += matrix1_in[ rows*size + j] * matrix2_in[j*size + cols];
    }
  }
}



/****************************************************
 
 ***************************************************/
void matrix_mult_sse(int size, double *matrix1_in,
		      double *matrix2_in, double *matrix_out){
  __m128d a_line, b_line, r_line;
  int i, j, jj;

// OpenMP is used to exploit thread-level parallelism in this function. To that effect, a loop work-sharing construct compiler directive was added to split the for loop iterations among the threads in a team, effectively assigning one thread to each of the vector by matrix multiplication problems we can split the application in. Furthermore the correct directives were given to specify the global and private variables of the implementation
# pragma omp parallel \
    shared(jj, matrix1_in, matrix2_in, matrix_out, size) \
    private(i , j, b_line, a_line, r_line)
# pragma omp for
    for(jj=0; jj<size; jj++){
        // Incrementing clause was halved since the data type size doubled and now only two double floating point data types are loaded with each mm_load_pd operation
        for (i=0; i<size; i+=2){
            j = 0;
            // _mm_load_ps, _mm_set1_ps and _mm_mul_ps changed to load two double floating point data types
            b_line = _mm_load_pd(&matrix2_in[i]); // b_line = vec4(matrix[i][0])
            a_line = _mm_set1_pd(matrix1_in[j+(jj*size)]);      // a_line = vec4(vector_in[0])
            r_line = _mm_mul_pd(a_line, b_line); // r_line = a_line * b_line
            for (j=1; j<size; j++) {
                b_line = _mm_load_pd(&matrix2_in[j*size +i]); // a_line = vec4(column(a, j))
                a_line = _mm_set1_pd(matrix1_in[j+(jj*size)]);  // b_line = vec4(b[i][j])
                // r_line += a_line * b_line
                r_line = _mm_add_pd(_mm_mul_pd(a_line, b_line), r_line);
            }
            _mm_store_pd(&matrix_out[jj*size + i], r_line);     // r[i] = r_line
        }
    }
}



/****************************************************
 
 ***************************************************/
int main(int argc, char *argv[]){
    
  int i, j, adsize;
  if(argc < 2){
    printf("Usage: %s matrix/vector_size\n", argv[0]);
    return 0;
  }

  // Calculates the lowest multiple of two that is greater than "size" and stores the value in adsize. Adsize will be size of padded matrix and "size" of the actual matrix. If size==adsize, no padding will be required
  int size = atoi(argv[1]);
    adsize = size;
  if(size%2 != 0){
    do{ adsize++; }while(adsize%2 != 0);
  }
    
  // Alocation of "adsize*adsize" number of double floating point data types (to store the matriz operands) with a memory address multiple of sizeof(double)*2. This last argument was halved as now our data types are twice as long
  double *matrix1 = (double *)memalign(sizeof(double)*2, sizeof(double)*adsize*adsize);
  if(matrix1==NULL){
    printf("can't allocate the required memory for matrix\n");
    free(matrix1);
    return 0;
  }

  double *matrix2 = (double *)memalign(sizeof(double)*2, sizeof(double)*adsize*adsize);
  if(matrix2==NULL){
    printf("can't allocate the required memory for matrix\n");
    free(matrix2);
    return 0;
  }

  double *result_sq = (double *)memalign(sizeof(double)*2, sizeof(double)*adsize*adsize);
  if(result_sq==NULL){
    printf("can't allocate the required memory for result_sq\n");
    free(matrix1);
    free(matrix2);
    return 0;
  }

  double *result_pl = (double *)memalign(sizeof(double)*2, sizeof(double)*adsize*adsize);
  if(result_pl==NULL){
    printf("can't allocate the required memory for result_pl\n");
    free(matrix1);
    free(matrix2);
    free(result_sq);
    return 0;
  }

  matrix_gen(size, adsize, matrix1);
  matrix_gen(size, adsize, matrix2);
    
  double time_sq;
  double time_sse;

  time_sq = omp_get_wtime();
  matrix_mult_sq(adsize, matrix1, matrix2, result_sq);
  time_sq = omp_get_wtime() - time_sq;

  time_sse = omp_get_wtime();
  matrix_mult_sse(adsize, matrix1, matrix2, result_pl);
  time_sse = omp_get_wtime() - time_sse;
    
  printf("SEQUENTIAL EXECUTION: %f (sec)\n",time_sq);
  printf("PARALLEL EXECUTION: %f (sec)\n", time_sse);
    
    
    // If padding was used, this if clause will take the resulting matrix and create the requested result. In effect it will now allocate a correct size structure and clean the artifacts created by the adding of the padding to the operand matrixes, copying the correct results to the final resulting matrix
    double *fresult_sq, *fresult_pl;
    if(size!=adsize){
        fresult_sq = (double *)memalign(sizeof(double)*2, sizeof(double)*size*size);
        fresult_pl = (double *)memalign(sizeof(double)*2, sizeof(double)*size*size);
        if(fresult_pl==NULL || fresult_sq==NULL){
            printf("can't allocate the required memory for result_pl\n");
            free(matrix1);
            free(matrix2);
            free(result_sq);
            return 0;
        }
        
        j=0;
        for(i=0;i<adsize*adsize; i++){
            if(  !( ((i+1)%adsize==0 && (adsize-size >0)) || i>(size*adsize-1) )  ){
                fresult_pl[j]= result_pl[i];
                fresult_sq[j]= result_sq[i];
                j++;
            }
        }
    }
    

  //check
  for(i=0; i<size*size; i++)
    if((int)result_sq[i] != (int)result_pl[i]){
      printf("wrong at position %d\n", i);
      free(matrix1);
      free(matrix2);
      free(result_sq);
      free(result_pl);
      return 0;
    }
    
  //printf("\nCorrect Result!\n\n");

  free(matrix1);
  free(matrix2);
  free(result_sq);
  free(result_pl);
  if(size!=adsize){
    free(fresult_sq);
    free(fresult_pl);
  }
    
  return 1;
}
