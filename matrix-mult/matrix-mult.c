/*
  Universidade Federal de São Carlos
  Departamento de Computação
  Alunos:          RA's:
  Caio Ueno        743516
  Gabriel Cheban   743535
  Prof. Hélio Crestana Guardia
  Programação Paralela e Distribuída
/

/
  Programa: Multiplicação de matrizes com CUDA
  Objetivo: Desenvolver um código para realização da multiplicação de matrizes em paralelo,
            usando a biblioteca para programação em CUDA.
  Link para github: https://github.com/CaioUeno/parallel-and-distributed-programming
*/

#include <stdio.h>
#define ROWS_1 1000
#define COLS_1 2000

#define ROWS_2 2000 
#define COLS_2 3000


__global__
void matrix_mult(float *m1, float *m2, float *m3) {
  
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;

  // Se a thread estiver dentro da matriz, realiza a operação
  if (row < ROWS_2 && col < COLS_1){
      m3[row*COLS_2+col] = 0;
      for(int i = 0; i < COLS_1; i++)
        m3[row*COLS_2+col] += m1[row*COLS_1+i] * m2[i*COLS_2+col];
  }

}

void print_matrix(float *m, int rows, int cols){
    for(int i = 0; i < rows * cols; i++){
        if(i % cols == 0)
            printf("\n");
        printf("%.2f ", m[i]);
    }
}

int maximum(int a, int b){
    if(a >= b)
        return a;
    return b;
}

int main(void) {

  // Verifica se as matrizes podem se multiplicadas
  if (COLS_1 != ROWS_2) {
      printf("Número de linhas da M2 tem que ser igual a número de colunas da M1\n");
      exit(0);
  }

  float *m1, *m2, *m3, *d_m1, *d_m2, *d_m3;

  // Alocação dinâmica das matrizes
  m1 = (float *) malloc(ROWS_1*COLS_1*sizeof(float *));
  m2 = (float *) malloc(ROWS_2*COLS_2*sizeof(float *));
  m3 = (float *) malloc(ROWS_1*COLS_2*sizeof(float *));

  for(int i = 0; i < ROWS_1*COLS_1; i++)
      m1[i] = i;
  
  for(int i = 0; i < ROWS_2*COLS_2; i++)
      m2[i] = i;


  // Alocação de memória na GPU
  cudaMalloc(&d_m1, ROWS_1*COLS_1*sizeof(float)); 
  cudaMalloc(&d_m2, ROWS_2*COLS_2*sizeof(float));
  cudaMalloc(&d_m3, ROWS_1*COLS_2*sizeof(float));

  // Cópia dos dados da memória RAM para a memória do dispositivo
  cudaMemcpy(d_m1, m1, ROWS_1*COLS_1*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m2, m2, ROWS_2*COLS_2*sizeof(float), cudaMemcpyHostToDevice);
  
  // Realiza a múltiplicação em paralelo
  dim3 threadsPerBlock(22, 22); // 22² ~ 512
  dim3 blocksPerGrid(ROWS_2, maximum(ROWS_1, COLS_2));
  matrix_mult<<<blocksPerGrid, threadsPerBlock>>>(d_m1, d_m2, d_m3);

  // Cópia dos dados da memória da GPU para a memória RAM
  cudaMemcpy(m3, d_m3, ROWS_1*COLS_2*sizeof(float), cudaMemcpyDeviceToHost);

/*
  // checando resultado
    printf("\n\nMatriz 1: \n");
  print_matrix(m1, ROWS_1, COLS_1);
  printf("\n\nMatriz 2: \n");
  print_matrix(m2, ROWS_2, COLS_2);
    printf("\n\nMatriz 3: \n");
  print_matrix(m3, ROWS_1, COLS_2);
*/

  // Liberação das áreas de memória alocadas da GPU
  cudaFree(d_m1);
  cudaFree(d_m2);
  cudaFree(d_m3);
  free(m1);
  free(m2);
  free(m3);
  d_m1 = NULL;
  d_m2 = NULL;
  d_m3 = NULL;
  m1 = NULL;
  m2 = NULL;
  m3 = NULL;

}