%%cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

// Define as condições de parada do algoritmo.
#define MAX_ITER 500
#define TOL 0.0001

// Define as quantidades de instâncias, características e grupos.
#define N_INSTANCES 1000
#define N_FEATURES 250
#define N_CLUSTERS 25

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

    // Aloca a matriz (como um vetor) de instâncias.
    km->instances = (double *) malloc(km->n_instances*km->n_features*sizeof(double));

    // Aloca dinamicamente as instâncias.
    for(int i = 0; i < km->n_instances; i++){

        // Atribui valores às features.
        for (int f = 0; f < km->n_features; f++)
            km->instances[i*km->n_features + f] = i; //sin(f+rand()%10)*cos(i+rand()%6)*(i+2)*pow(-1,rand()%2); Versão aleatória.
    }

    // Aloca dinamicamente o vetor de rótulos.
    km->labels = (int *) malloc(km->n_instances*sizeof(int));
}

//------------------------------------------------------------------------------

void select_centroids(k_means *km){

    /*
        Função que seleciona os centroides da primeira iteração.
        Versão 1D.
    */

    // Aloca dinamicamente a matriz (como um vetor) de centroides.
    km->centroids = (double *) malloc(km->n_clusters*km->n_features*sizeof(double));

    // Atribui valores às features.
    for(int c = 0; c < km->n_clusters; c++)
        for (int f = 0; f < km->n_features; f++)
            km->centroids[c*km->n_features + f] = km->instances[c*km->n_features + f];

    // Aloca dinamicamente o vetor de deslocamentos dos centroides.
    km->displacement = (double *) calloc(km->n_clusters, sizeof(double));
}

__global__
void label_instances(double *inst, double *cent, int *labs){

    /*
        Função (somente na GPU) que atribui a cada instância o rótulo do centroide mais próximo.
    */

    // Define qual instância será calculada.
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    double current_dist, min_dist;

    // Verifica se está no escopo de instâncias.
    if (i < N_INSTANCES){
        for (int c = 0; c < N_CLUSTERS; c++){
            current_dist = 0;

            // Calcula a distância euclidiana entre a instância e o centroide corrente.
            for (int f = 0; f < N_FEATURES; f++)
                current_dist += pow((cent[c*N_FEATURES+f] - inst[i*N_FEATURES+f]), 2);
            current_dist = sqrt(current_dist);

            // Atribui como distância mínima, caso seja a primeira iteração.
            if(c == 0){
                min_dist = current_dist;
                labs[i] = c;
            }

            // Atualiza a distância mínima.
            if(current_dist < min_dist){
                min_dist = current_dist;
                labs[i] = c;
            }
        }
    }
}

__global__
void update_centroids(double *inst, double *cent, int *labs, double *disp){

    /*
        Função (somente na GPu) que atualiza cada centroide.
    */

    // Define qual centroide será calculado.
    int c = blockIdx.x*blockDim.x+threadIdx.x;

    // Verifica se está no escopo de centroides.
    if (c < N_CLUSTERS){
        int counter;
        double feature_sum, current_delta = 0, mean_deltas = 0;

        for (int f = 0; f < N_FEATURES; f++){
            counter = 0;
            feature_sum = 0;

            // Soma a feature corrente de todas as instâncias que pertencem ao centroide corrente e as conta.
            for (int i = 0; i < N_INSTANCES; i++)
                if(labs[i] == c){
                    counter++;
                    feature_sum += inst[i*N_FEATURES + f];
                }

            // Calcula o deslocamento dinamicamente (ao longo das iterações).
            current_delta += pow(cent[c*N_FEATURES + f] - feature_sum/counter, 2);

            // Atualiza a feature (dimensão) corrente do centroide corrente.
            cent[c*N_FEATURES + f] = feature_sum/counter;
        }

        // Finaliza o cálculo da distância euclidiana entre o centroide antigo e atualizado.
        mean_deltas += sqrt(current_delta);

        // Armazena o deslocamento no vetor disp.
        disp[c] = mean_deltas;
    }
}
//------------------------------------------------------------------------------

void print_instances(k_means *km){

    /*
        Função que imprime as instâncias.
        Versão 1D.
    */

    printf("Instâncias: \n");

    for (int i = 0; i < km->n_instances*km->n_features; i++){
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

    printf("Centroides: \n");

    for (int c = 0; c < km->n_clusters*km->n_features; c++){
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

    for (int i = 0; i < km->n_instances*km->n_features; i++){
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

    for (int c = 0; c < km->n_clusters*km->n_features; c++){
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
        Função que desaloca as váriaveis dinâmicas da struct k_means.
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


int main(int argc, char const *argv[]){

    // Instancia uma struct do tipo k-means.
    k_means km;
    km.n_instances = N_INSTANCES;
    km.n_features = N_FEATURES;
    km.n_clusters = N_CLUSTERS;

    create_artificial_k_means(&km);
    select_centroids(&km);

    double mean_deltas;
    int iter = 0;

    // Variáveis para medida do tempo.
	struct timeval inic, fim;
    struct rusage r1, r2;

    // Obtém tempo e consumo de CPU antes de executar o algoritmo k-means.
	gettimeofday(&inic, 0);
    getrusage(RUSAGE_SELF, &r1);

    // Variáveis para GPU.
    double *gpu_instances, *gpu_centroids, *gpu_displacement;
    int *gpu_labels;
    int n_threads, n_blocks;
    dim3 threadsPerBlock, blocksPerGrid;

    // Alocação de memória na GPU.
    cudaMalloc(&gpu_instances, km.n_instances*km.n_features*sizeof(double));
    cudaMalloc(&gpu_centroids, km.n_clusters*km.n_features*sizeof(double));
    cudaMalloc(&gpu_labels, km.n_instances*sizeof(int));
    cudaMalloc(&gpu_displacement, km.n_clusters*sizeof(double));

    // Cópia dos dados da memória RAM para a memória do dispositivo.
    cudaMemcpy(gpu_instances, km.instances, km.n_instances*km.n_features*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroids, km.centroids, km.n_clusters*km.n_features*sizeof(double), cudaMemcpyHostToDevice);

    // Rotula as instâncias e atualiza os centroides até satisfazer uma das condições (MAX_ITER ou TOL).
    do{
        mean_deltas = 0;
        iter++;

        // Paralelização 1: Definição do rótulo das instâncias.
        if(km.n_instances <= 512)
            n_threads = km.n_instances;
        else
            n_threads = 512;
        threadsPerBlock = n_threads;

        n_blocks = ceil(km.n_instances/(float) n_threads);
        blocksPerGrid = n_blocks;

        label_instances<<<blocksPerGrid, threadsPerBlock>>>(gpu_instances, gpu_centroids, gpu_labels);

        // Paralelização 2: Atualização dos centroides.
        if(km.n_clusters <= 512)
            n_threads = km.n_clusters;
        else
            n_threads = 512;
        threadsPerBlock = n_threads;

        n_blocks = ceil(km.n_clusters/(float) n_threads);
        blocksPerGrid = n_blocks;

        update_centroids<<<blocksPerGrid, threadsPerBlock>>>(gpu_instances, gpu_centroids, gpu_labels, gpu_displacement);

        cudaMemcpy(km.displacement, gpu_displacement, km.n_clusters*sizeof(double), cudaMemcpyDeviceToHost);

        // Calcula a média dos deslocamentos (em CPU).
        for (int c = 0; c < km.n_clusters; c++)
            mean_deltas += km.displacement[c];
        mean_deltas /= km.n_clusters;

        printf("Iteração: %d; Delta: %lf\n", iter, mean_deltas);

    } while(iter < MAX_ITER && mean_deltas > TOL);

    // Obtém tempo e consumo de CPU após executar o algoritmo k-means (utilizando GPU).
	gettimeofday(&fim,0);
	getrusage(RUSAGE_SELF, &r2);

	printf("\nElapsed time:%f sec\tUser time:%f sec\tSystem time:%f sec\n",
	 (fim.tv_sec+fim.tv_usec/1000000.) - (inic.tv_sec+inic.tv_usec/1000000.),
	 (r2.ru_utime.tv_sec+r2.ru_utime.tv_usec/1000000.) - (r1.ru_utime.tv_sec+r1.ru_utime.tv_usec/1000000.),
	 (r2.ru_stime.tv_sec+r2.ru_stime.tv_usec/1000000.) - (r1.ru_stime.tv_sec+r1.ru_stime.tv_usec/1000000.));

    // Cópia dos dados da memória da GPU para a memória RAM.
    cudaMemcpy(km.instances, gpu_instances, km.n_instances*km.n_features*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(km.centroids, gpu_centroids, km.n_clusters*km.n_features*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(km.labels, gpu_labels, km.n_instances*sizeof(int), cudaMemcpyDeviceToHost);

    // Prints para a depuração
    // print_labels(&km);
    // print_centroids(&km);

    // Armazena os resultados em arquivo (.txt).
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
