%%cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

// Define a condição de parada do algoritmo
#define MAX_ITER 500
#define TOL 0.0001

// Define as quantidades de instâncias, características e grupos
#define N_INSTANCES 20000
#define N_FEATURES 500
#define N_CLUSTERS 7

// Tipo de dado: k_means
typedef struct{
    int n_instances, n_features, n_clusters;
    double *instances, *centroids;
    int *labels;
    double *displacement;
} k_means;



//------------------------------------------------------------------------------

void create_artificial_k_means(k_means *km){

    /*
        Função que cria um dataset artificial.
        Versão 1D.
    */

    // Instanciando a matriz(vetor) de instâncias.
    km->instances = (double *) malloc(km->n_instances*km->n_features*sizeof(double));

    for(int i = 0; i < km->n_instances; i++){

        // Atribuindo valores as features
        for (int f = 0; f < km->n_features; f++)
            km->instances[i*km->n_features + f] = i; //sin(f+rand()%10)*cos(i+rand()%6)*(i+2)*pow(-1,rand()%2);
    }

    km->labels = (int *) malloc(km->n_instances*sizeof(int));
}

//------------------------------------------------------------------------------

void select_centroids(k_means *km){

    /*
        Função que seleciona os centroides da primeira iteração.
        Versão 1D.
    */

    // Instanciando a matriz de centroides.
    km->centroids = (double *) malloc(km->n_clusters*km->n_features*sizeof(double));

    for(int c = 0; c < km->n_clusters; c++)

        // Atribuindo valores as features
        for (int f = 0; f < km->n_features; f++)
            km->centroids[c*km->n_features + f] = km->instances[c*km->n_features + f];

    km->displacement = (double *) calloc(km->n_clusters, sizeof(double));
}

__global__
void label_instances(double *inst, double *cent, int *labs) {

    int i = blockIdx.x*blockDim.x+threadIdx.x;

    double current_dist, min_dist;

    if (i < N_INSTANCES){
        for (int c = 0; c < N_CLUSTERS; c++){

            current_dist = 0;
            for (int f = 0; f < N_FEATURES; f++)
                current_dist += pow((cent[c*N_FEATURES+f] - inst[i*N_FEATURES+f]), 2);
            current_dist = sqrt(current_dist);

            if(c == 0){
                min_dist = current_dist;
                labs[i] = c;
            }

            if(current_dist < min_dist){
                min_dist = current_dist;
                labs[i] = c;
            }
        }
    }
}

__global__
void update_centroids(double *inst, double *cent, int *labs, double *disp) {

    int c = blockIdx.x*blockDim.x+threadIdx.x;

    if (c < N_CLUSTERS) {
        int counter;
        double aux, current_delta, mean_deltas = 0;

        current_delta = 0;
        for (int f = 0; f < N_FEATURES; f++){
            counter = 0;
            aux = 0;
            for (int i = 0; i < N_INSTANCES; i++){
                if(labs[i] == c){
                    counter++;
                    aux += inst[i*N_FEATURES + f];
                }
            }
            current_delta += pow(cent[c*N_FEATURES + f] - aux/counter, 2);
            cent[c*N_FEATURES + f] = aux/counter;
        }
        mean_deltas += sqrt(current_delta);

        disp[c] = mean_deltas/N_CLUSTERS;
    }
}
//------------------------------------------------------------------------------

void print_instances(k_means *km){

    /*
        Função que imprime as instâncias.
        Versão 1D.
    */

    printf("Instâncias: \n");
    for (int i = 0; i < km->n_instances*km->n_features; i++) {
        if (i % km->n_features == 0)
            printf("\n");
        printf("%lf ", km->instances[i]);
    }
    printf("\n");
}

void print_centroids(k_means *km){

    /*
        Função que imprime os centroides.
        Versão 1D.
    */

    if (km->n_clusters == 0) {
        printf("É necessário definir os clusters antes de printá-los!\n");
        exit(0);
    }

    printf("Centroides: \n");
    for (int c = 0; c < km->n_clusters*km->n_features; c++) {
        if (c % km->n_features == 0)
            printf("\n");
        printf("%lf ", km->centroids[c]);

    }
    printf("\n");
}

void print_labels(k_means *km){

    /*
        Função que imprime os rótulos.
    */

    printf("Rótulos: \n");
    for (int i = 0; i < km->n_instances; i++)
        printf("%d \n", km->labels[i]);

    printf("\n");
}

//------------------------------------------------------------------------------

void save_instances(k_means *km){

    /*
        Função que imprime as instâncias.
    */

    FILE *arq;

    arq = fopen("instances.txt", "w");

    for (int i = 0; i < km->n_instances*km->n_features; i++) {
        if(i % km->n_features == 0 && i != 0)
            fprintf(arq, "\n");
        fprintf(arq, "%lf ", km->instances[i]);
    }
    fclose(arq);
}

void save_centroids(k_means *km){

    /*
        Função que salva os centroides.
    */

    FILE *arq;

    arq = fopen("centroides.txt", "w");

    for (int c = 0; c < km->n_clusters*km->n_features; c++) {
        if(c % km->n_features == 0 && c != 0)
            fprintf(arq, "\n");
        fprintf(arq, "%lf ", km->centroids[c]);
    }

    fclose(arq);
}

void save_labels(k_means *km){

    /*
        Função que salve os rótulos.
    */
    FILE *arq;
    arq = fopen("labels.txt", "w");
    for (int i = 0; i < km->n_instances; i++)
        fprintf(arq, "%d\n", km->labels[i]);
    fclose(arq);
}

//------------------------------------------------------------------------------

void free_k_means(k_means *km){

    /*
        Função que desaloca o vetor de structs.
    */

    free(km->instances);
    free(km->centroids);
    free(km->labels);
    free(km->displacement);
    km->instances = NULL;
    km->centroids = NULL;
    km->labels = NULL;
    km->displacement = NULL;
}

//------------------------------------------------------------------------------


int main(int argc, char const *argv[]) {

    k_means km;
    km.n_instances = N_INSTANCES;
    km.n_features = N_FEATURES;
    km.n_clusters = N_CLUSTERS;

    double mean_deltas;
    int iter = 0;

    create_artificial_k_means(&km);
    select_centroids(&km);

    // Variáveis para medida do tempo
	struct timeval inic, fim;
    struct rusage r1, r2;

    // Obtém tempo e consumo de CPU antes da aplicação do filtro
	gettimeofday(&inic, 0);
    getrusage(RUSAGE_SELF, &r1);

    // Variáveis para GPU
    double *gpu_instances, *gpu_centroids, *gpu_displacement;
    int *gpu_labels;
    int n_threads, n_blocks;

    // Alocação de memória na GPU
    cudaMalloc(&gpu_instances, km.n_instances*km.n_features*sizeof(double));
    cudaMalloc(&gpu_centroids, km.n_clusters*km.n_features*sizeof(double));
    cudaMalloc(&gpu_labels, km.n_instances*sizeof(int));
    cudaMalloc(&gpu_displacement, km.n_clusters*sizeof(double));

    // Cópia dos dados da memória RAM para a memória do dispositivo
    cudaMemcpy(gpu_instances, km.instances, km.n_instances*km.n_features*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroids, km.centroids, km.n_clusters*km.n_features*sizeof(double), cudaMemcpyHostToDevice);

    do {
        iter++;

        // Paralelização 1: Definição do rótulo das instâncias
        if(km.n_instances <= 512)
            n_threads = km.n_instances;
        else
            n_threads = 512;

        dim3 threadsPerBlock(n_threads);

        n_blocks = ceil(km.n_instances/(float) n_threads);

        dim3 blocksPerGrid(n_blocks);
        label_instances<<<blocksPerGrid, threadsPerBlock>>>(gpu_instances, gpu_centroids, gpu_labels);

        // Paralelização 2: Atualização dos centroides
        if(km.n_clusters <= 512)
            n_threads = km.n_clusters;
        else
            n_threads = 512;

        threadsPerBlock = n_threads;

        n_blocks = ceil(km.n_clusters/(float) n_threads);

        blocksPerGrid = n_blocks;

        update_centroids<<<blocksPerGrid, threadsPerBlock>>>(gpu_instances, gpu_centroids, gpu_labels, gpu_displacement);
        cudaMemcpy(km.displacement, gpu_displacement, km.n_clusters*sizeof(double), cudaMemcpyDeviceToHost);

        for (int c = 0; c < km.n_clusters; c++)
            mean_deltas += km.displacement[c];

        mean_deltas /= km.n_clusters;

        printf("Iteração: %d; Delta: %lf\n", iter, mean_deltas);

    } while(iter < MAX_ITER && mean_deltas > TOL);

    // obtém tempo e consumo de CPU depois da aplicação do filtro
	gettimeofday(&fim,0);
	getrusage(RUSAGE_SELF, &r2);

	printf("\nElapsed time:%f sec\tUser time:%f sec\tSystem time:%f sec\n",
	 (fim.tv_sec+fim.tv_usec/1000000.) - (inic.tv_sec+inic.tv_usec/1000000.),
	 (r2.ru_utime.tv_sec+r2.ru_utime.tv_usec/1000000.) - (r1.ru_utime.tv_sec+r1.ru_utime.tv_usec/1000000.),
	 (r2.ru_stime.tv_sec+r2.ru_stime.tv_usec/1000000.) - (r1.ru_stime.tv_sec+r1.ru_stime.tv_usec/1000000.));

    // Cópia dos dados da memória da GPU para a memória RAM
    cudaMemcpy(km.instances, gpu_instances, km.n_instances*km.n_features*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(km.centroids, gpu_centroids, km.n_clusters*km.n_features*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(km.labels, gpu_labels, km.n_instances*sizeof(int), cudaMemcpyDeviceToHost);

    // Prints para a depuração
    // print_labels(&km);
    // print_centroids(&km);

    save_instances(&km);
    save_centroids(&km);
    save_labels(&km);

    free_k_means(&km);
    cudaFree(gpu_instances);
    cudaFree(gpu_centroids);
    cudaFree(gpu_displacement);
    cudaFree(gpu_labels);
    gpu_instances = NULL;
    gpu_centroids = NULL;
    gpu_displacement = NULL;
    gpu_labels = NULL;

    return 0;
}
