#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "mpi.h"

// Define as condições de parada do algoritmo.
#define MAX_ITER 500
#define TOL 0.0001

// Define as quantidades de instâncias, características e grupos.
#define N_INSTANCES 10000
#define N_FEATURES 500
#define N_CLUSTERS 250

typedef struct{
    int n_instances, n_features, n_clusters;
    double **instances, **centroids;
    int *labels, *cluster_count;
} k_means;

//------------------------------------------------------------------------------

void create_artificial_k_means(k_means *km){

    /*
        Função que cria um dataset artificial.
    */

    // Aloca a matriz de instâncias.
    km->instances = (double **) malloc(km->n_instances*sizeof(double *));

    // Aloca dinamicamente as instâncias.
    for(int i = 0; i < km->n_instances; i++){
        km->instances[i] = (double *) malloc(km->n_features*sizeof(double));

        // Atribui valores às features.
        for (int f = 0; f < km->n_features; f++)
            km->instances[i][f] = i; //sin(f+rand()%10)*cos(i+rand()%6)*(i+2)*pow(-1,rand()%2); Versão aleatória.
    }

    // Aloca dinamicamente o vetor de rótulos.
    km->labels = (int *) malloc(km->n_instances*sizeof(int));
}

//------------------------------------------------------------------------------

void select_centroids(k_means *km){

    /*
        Função que seleciona os centroides da primeira iteração.
    */

    // Aloca a matriz de centroides.
    km->centroids = (double **) malloc(km->n_clusters*sizeof(double *));

    // Aloca dinamicamente os centroides.
    for(int c = 0; c < km->n_clusters; c++){
        km->centroids[c] = (double *) malloc(km->n_features*sizeof(double));

        // Atribui valores às features.
        for (int f = 0; f < km->n_features; f++)
            km->centroids[c][f] = km->instances[c][f];
    }

    // Aloca dinamicamente o vetor para contagem de instâncias por centroide.
    km->cluster_count = (int *) malloc(km->n_clusters*sizeof(int));
}

void counter(k_means *km){

    /*
        Função que conta quantas instâncias cada centroide possui.
    */

    // Zera o vetor de contagem de instâncias.
    for (int c = 0; c < km->n_clusters; c++)
        km->cluster_count[c] = 0;

    // Conta quantas instâncias pertencem a cada rótulo.
    for (int i = 0; i < km->n_instances; i++)
        km->cluster_count[km->labels[i]]++;
}

int nearest_centroid_id(double *inst, double **cents){

    /*
        Função que retorna o rótulo do centroide mais próximo da instância dada.
    */

    int min_index;
    double current_dist, min_dist;
    MPI_Status status;

    for (int c = 0; c < N_CLUSTERS; c++){
        current_dist = 0;

        // Calcula a distância euclidiana entre a instância e o centroide corrente.
        for (int f = 0; f < N_FEATURES; f++)
            current_dist += pow((cents[c][f] - inst[f]), 2);
        current_dist = sqrt(current_dist);

        // Atribui como distância mínima, caso seja a primeira iteração.
        if(c == 0){
            min_dist = current_dist;
            min_index = c;
        }

        // Atualiza a distância mínima.
        if(current_dist < min_dist){
            min_dist = current_dist;
            min_index = c;
        }
    }

    return min_index;
}

//------------------------------------------------------------------------------

void print_instances(k_means *km){

    /*
        Função que imprime as instâncias.
    */

    printf("Instâncias: \n");

    for (int i = 0; i < km->n_instances; i++){
        for (int f = 0; f < km->n_features; f++)
            printf("%lf ", km->instances[i][f]);

        printf("\n");
    }

    printf("\n");
}

void print_centroids(k_means *km){

    /*
        Função que imprime os centroides.
    */

    printf("Centroides: \n");

    for (int c = 0; c < km->n_clusters; c++){
        for (int f = 0; f < km->n_features; f++)
            printf("%lf ", km->centroids[c][f]);

        printf("\n");
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

    for (int i = 0; i < km->n_instances; i++){
        for (int f = 0; f < km->n_features; f++)
            fprintf(arq, "%lf ", km->instances[i][f]);

        fprintf(arq, "\n");
    }

    fclose(arq);
}

void save_centroids(k_means *km){

    /*
        Função que salva os centroides.
    */

    FILE *arq;

    arq = fopen("centroides.txt", "w");

    for (int c = 0; c < km->n_clusters; c++){
        for (int f = 0; f < km->n_features; f++)
            fprintf(arq, "%lf ", km->centroids[c][f]);

        fprintf(arq, "\n");
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

    for(int i = 0; i < km->n_instances; i++){
        free(km->instances[i]);
        km->instances[i] = NULL;
    }

    for(int c = 0; c < km->n_clusters; c++){
        free(km->centroids[c]);
        km->centroids[c] = NULL;
    }

    free(km->instances);
    free(km->centroids);
    free(km->labels);
    free(km->cluster_count);
    km->instances = NULL;
    km->centroids = NULL;
    km->labels = NULL;
    km->cluster_count = NULL;
}

//------------------------------------------------------------------------------

int main(int argc, char **argv){

    // Instancia as variáveis para o MPI.
    int rank, size, n_process, type;
    MPI_Status status;

    // Inicializa MPI.
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Instancia uma struct do tipo k-means e variáveis.
    k_means km;
    double mean_deltas, current_delta;
    double *instance, *centroid, *new_centroid, **centroids;
    int iter = 0;

    // Variáveis auxiliares para enviar instâncias e centroides como mensagens.
    instance = (double *) malloc(N_FEATURES*sizeof(double));
    centroid = (double *) malloc(N_FEATURES*sizeof(double));
    new_centroid = (double *) malloc(N_FEATURES*sizeof(double));
    centroids = (double **) malloc(N_CLUSTERS*sizeof(double *));
    for (int c = 0; c < N_CLUSTERS; c++)
        centroids[c] = (double *) malloc(N_FEATURES*sizeof(double));

    int label, count;


    // Somente o processo mestre possui os dados.
    if (rank == 0){

        km.n_instances = N_INSTANCES;
        km.n_features = N_FEATURES;
        km.n_clusters = N_CLUSTERS;

        create_artificial_k_means(&km);
        select_centroids(&km);

    }

    // Variáveis para medida do tempo.
	struct timeval inic, fim;
    struct rusage r1, r2;

    // Obtém tempo e consumo de CPU antes de executar o algoritmo k-means.
	gettimeofday(&inic, 0);
    getrusage(RUSAGE_SELF, &r1);

    // Rotula as instâncias e atualiza os centroides até satisfazer uma das condições (MAX_ITER ou TOL) por troca de mensagens.
    do{
        iter++;

        // Trecho de código do processo mestre.
        if (rank == 0){
            mean_deltas = 0;

            // Manda cada um dos centroides para todos os trabalhadores.
            for(int p = 1; p < size; p++)
                for (int c = 0; c < km.n_clusters; c++){

                    // Coloca o centroide num vetor auxiliar.
                    for (int f = 0; f < km.n_features; f++)
                    centroid[f] = km.centroids[c][f];

                    // Manda o centroide.
                    MPI_Send(&centroid[0], N_FEATURES, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);
                }

            // Laço para rotular instâncias.
            for(int i = 0; i < km.n_instances; i += size-1){
                for(int p = 1; p < size; p++){

                    if(i+p-1 < km.n_instances){

                        // Coloca a instância em um vetor auxiliar.
                        for (int f = 0; f < km.n_features; f++)
                            instance[f] = km.instances[i+p-1][f];

                        // Manda a instância corrente para o trabalhador p.
                        MPI_Send(&instance[0], km.n_features, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);
                    }
                }

                for(int p = 1; p < size; p++)
                    if(i+p-1 < km.n_instances){
                        // Recebe do trabalhador p o rótulo do centroide mais próximo da instância enviada anteriormente.
                        MPI_Recv(&label, 1, MPI_INT, p, 99, MPI_COMM_WORLD, &status);

                        // Atribui o rótulo no vetor de rótulos.
                        km.labels[i+p-1] = label;
                    }
            }

            // Define a quantidade de instâncias por centroide.
            counter(&km);

            // Laço para atualizar os centroides.
            for(int c = 0; c < km.n_clusters; c += size-1){
                for(int p = 1; p < size; p++)
                    if(c+p-1 < km.n_clusters){

                        // Coloca o centroide em um vetor auxiliar.
                        for (int f = 0; f < km.n_features; f++)
                            centroid[f] = km.centroids[c+p-1][f];

                        // Manda o centroide corrent para o trabalhador p.
                        MPI_Send(&centroid[0], N_FEATURES, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);

                        // Manda quantas instâncias o centroide corrente possui.
                        MPI_Send(&km.cluster_count[c+p-1], 1, MPI_INT, p, 99, MPI_COMM_WORLD);

                        // Manda todas as instâncias que o centroide corrente possui para o trabalhador p.
                        for (int i = 0; i < km.n_instances; i++)
                            if (km.labels[i] == c+p-1){

                                // Coloca a instância num vetor auxiliar.
                                for (int f = 0; f < km.n_features; f++)
                                    instance[f] = km.instances[i][f];

                                // Manda a instância.
                                MPI_Send(&instance[0], km.n_features, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);
                            }
                    }

                for(int p = 1; p < size; p++)
                    if(c+p-1 < km.n_clusters){

                        // Recebe centroide corrente atualizado do processo p.
                        MPI_Recv(&centroid[0], N_FEATURES, MPI_DOUBLE, p, 99, MPI_COMM_WORLD, &status);

                        // Recebe o delta corrente do processo p.
                        MPI_Recv(&current_delta, 1, MPI_DOUBLE, p, 99, MPI_COMM_WORLD, &status);

                        // Coloca a centroide em um vetor auxiliar.
                        for (int f = 0; f < km.n_features; f++)
                            km.centroids[c+p-1][f] = centroid[f];

                        // Somatório dos deslocamentos.
                        mean_deltas += current_delta;
                    }
            }

            // Calcula a média dos deslocamentos.
            mean_deltas /= km.n_clusters;

            // Manda para todos os trabalhadores a condição de parada.
            for(int p = 1; p < size; p++)
                MPI_Send(&mean_deltas, 1, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);


        }

        // Trecho de código dos processos trabalhadores.
        else{

            for(int c = 0; c < N_CLUSTERS; c++){

                // Recebe do mestre o vetor que contém o centroide corrente.
                MPI_Recv(&centroid[0], N_FEATURES, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);

                for(int f = 0; f < N_FEATURES; f++)
                    centroids[c][f] = centroid[f];
            }

            // Laço para rotular as instâncias.
            for(int i = rank-1; i < N_INSTANCES; i += size-1){

                // Recebe do mestre uma instância.
                MPI_Recv(&instance[0], N_FEATURES, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);

                label = nearest_centroid_id(instance, centroids);

                // Manda para o mestre o rótulo do centroide mais próximo da instância recebida.
                MPI_Send(&label, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
            }

            // Laço para atualizar os centroides.
            for(int c = rank-1; c < N_CLUSTERS; c += size-1){

                // Zera o vetor.
                for (int f = 0; f < N_FEATURES; f++)
                    new_centroid[f] = 0;

                current_delta = 0;

                // Recebe do mestre um centroide.
                MPI_Recv(&centroid[0], N_FEATURES, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);

                // Recebe do mestre a contagem de instâncias do centroide corrente.
                MPI_Recv(&count, 1, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);

                for(int i = 0; i < count; i++){

                    // Recebe do mestre cada instância.
                    MPI_Recv(&instance[0], N_FEATURES, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);

                    // Calcula o novo centroide dinamicamente (ao longo das iterações).
                    for (int f = 0; f < N_FEATURES; f++)
                        new_centroid[f] += instance[f]/count;
                }

                // Calcula deslocamento do centroide corrente.
                for (int f = 0; f < N_FEATURES; f++)
                    current_delta += pow(centroid[f] - new_centroid[f], 2);

                // Finaliza o cálculo da distância euclidiana entre o centroide antigo e atualizado.
                current_delta = sqrt(current_delta);

                // Manda o novo centroide para o mestre.
                MPI_Send(&new_centroid[0], N_FEATURES, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);

                // Manda o deslocalmento para o mestre.
                MPI_Send(&current_delta, 1, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
            }

            // Recebe do mestre a condição de parada.
            MPI_Recv(&mean_deltas, 1, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);
        }

        if (rank == 0)
            printf("Iteração: %d; Delta: %lf\n", iter, mean_deltas);

    } while(iter < MAX_ITER && mean_deltas > TOL);

    // Processo mestre salva e desaloca k-means.
    if(rank == 0){

        // Obtém tempo e consumo de CPU após executar o algoritmo k-means (utilizando MPI).
        gettimeofday(&fim,0);
    	getrusage(RUSAGE_SELF, &r2);

    	printf("\nElapsed time:%f sec\tUser time:%f sec\tSystem time:%f sec\n",
    	 (fim.tv_sec+fim.tv_usec/1000000.) - (inic.tv_sec+inic.tv_usec/1000000.),
    	 (r2.ru_utime.tv_sec+r2.ru_utime.tv_usec/1000000.) - (r1.ru_utime.tv_sec+r1.ru_utime.tv_usec/1000000.),
    	 (r2.ru_stime.tv_sec+r2.ru_stime.tv_usec/1000000.) - (r1.ru_stime.tv_sec+r1.ru_stime.tv_usec/1000000.));

        // Prints para a depuração
        // print_labels(&km);
        // print_centroids(&km);

        save_instances(&km);
        save_centroids(&km);
        save_labels(&km);

        free_k_means(&km);
    }

    MPI_Finalize();

    for(int c = 0; c < N_CLUSTERS; c++){
        free(centroids[c]);
        centroids[c] = NULL;
    }
    free(centroids);
    free(instance);
    free(centroid);
    free(new_centroid);
    centroids = NULL;
    instance = NULL;
    centroid = NULL;
    new_centroid = NULL;

    return 0;
}
