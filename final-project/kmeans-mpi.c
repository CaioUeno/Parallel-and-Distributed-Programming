#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "mpi.h"

// Define a condição de parada do algoritmo
#define MAX_ITER 10
#define TOL 0.0001

// Define as quantidades de instâncias, características e grupos
#define N_INSTANCES 100
#define N_FEATURES 2
#define N_CLUSTERS 7

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

    // Instanciando a matriz de instâncias.
    km->instances = (double **) malloc(km->n_instances*sizeof(double));

    for(int i = 0; i < km->n_instances; i++){

        // Alocando dinamicamente uma instância
        km->instances[i] = (double *) malloc(km->n_features*sizeof(double));

        // Atribuindo valores as features
        for (int f = 0; f < km->n_features; f++)
            km->instances[i][f] = sin(f+rand()%10)*cos(i+rand()%6)*(i+2)*pow(-1,rand()%2); //sin(f+rand()%10)*cos(i+rand()%6)*(i+2)*pow(-1,rand()%2);
    }

    km->labels = (int *) malloc(km->n_instances*sizeof(int));

}

//------------------------------------------------------------------------------

void select_centroids(k_means *km){

    /*
        Função que seleciona os centroides da primeira iteração.
    */

    // Instanciando a matriz de centroides.
    km->centroids = (double **) malloc(km->n_clusters*sizeof(double));

    for(int c = 0; c < km->n_clusters; c++){

        // Alocando dinamicamente um centroide
        km->centroids[c] = (double *) malloc(km->n_features*sizeof(double));

        // Atribuindo valores as features
        for (int f = 0; f < km->n_features; f++)
            km->centroids[c][f] = km->instances[c][f];
    }

    km->cluster_count = (int *) malloc(km->n_clusters*sizeof(int));
}

void counter(k_means *km){

    // Zerando o vetor de contagem de instâncias.
    for (int c = 0; c < km->n_clusters; c++)
        km->cluster_count[c] = 0;

    // Contando quantas instâncias pertencem a cada rótulo.
    for (int i = 0; i < km->n_instances; i++)
        km->cluster_count[km->labels[i]]++;
}

int nearest_centroid_id(double *inst, double *cent){

    /*
        Função que retorna o rótulo do centroide mais perto da instância dada.
    */

    int min_index;
    double current_dist, min_dist;
    MPI_Status status;

    for (int c = 0; c < N_CLUSTERS; c++){

        // Recebendo do mestre o vetor contendo o centroide corrente.
        MPI_Recv(&cent[0], N_FEATURES, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);
        // printf("Recebi o centroid. Mas qual? Não sei pepposad.\n");

        current_dist = 0;
        for (int f = 0; f < N_FEATURES; f++)
            current_dist += pow((cent[f] - inst[f]), 2);

        current_dist = sqrt(current_dist);

        if(c == 0){
            min_dist = current_dist;
            min_index = c;
        }

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
    for (int i = 0; i < km->n_instances; i++) {
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

    if (km->n_clusters == 0) {
        printf("É necessário definir os clusters antes de printá-los!\n");
        exit(0);
    }

    printf("Centroides: \n");
    for (int c = 0; c < km->n_clusters; c++) {
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

    for (int i = 0; i < km->n_instances; i++) {
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

    for (int c = 0; c < km->n_clusters; c++) {
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
        Função que desaloca o vetor de structs.
    */

    for(int i = 0; i < km->n_instances; i++){
        free(km->instances[i]);
        km->instances[i] = NULL;
    }

    for(int i = 0; i < km->n_clusters; i++){
        free(km->centroids[i]);
        km->centroids[i] = NULL;
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

int main(int argc, char **argv) {

    int rank, size, n_process, type;
    MPI_Status status;

    // Inicializa MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    k_means km;
    double mean_deltas, current_delta;
    double *instance, *centroid, *new_centroid;
    int iter = 0;

    // Variáveis auxiliares para enviar instâncias e centroides como mensagens.
    instance = (double *) malloc(N_FEATURES*sizeof(double));
    centroid = (double *) malloc(N_FEATURES*sizeof(double));
    new_centroid = (double *) malloc(N_FEATURES*sizeof(double));
    int label, count;

    if (rank == 0) {
        // Somente o processo mestre possui os dados.
        km.n_instances = N_INSTANCES;
        km.n_features = N_FEATURES;
        km.n_clusters = N_CLUSTERS;

        create_artificial_k_means(&km);
        select_centroids(&km);
    }

    // variáveis para medida do tempo
	struct timeval inic, fim;
    struct rusage r1, r2;

    // obtém tempo e consumo de CPU antes da aplicação do filtro
	gettimeofday(&inic, 0);
    getrusage(RUSAGE_SELF, &r1);


    do {
        iter++;

        // Mestre
        if (rank == 0) {

            mean_deltas = 0;

            // Laço para rotular instâncias.
            for(int i = 0; i < km.n_instances; i += size-1){
                for(int p = 1; p < size; p++){

                    if(i+p-1 < km.n_instances){

                        // Colocando a instância num vetor auxiliar.
                        for (int f = 0; f < km.n_features; f++)
                            instance[f] = km.instances[i+p-1][f];

                        // Mandando a instância.
                        MPI_Send(&instance[0], km.n_features, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);

                        // Mandando cada um dos centroides.
                        for (int c = 0; c < km.n_clusters; c++) {

                            // Colocando o centroide num vetor auxiliar.
                            for (int f = 0; f < km.n_features; f++)
                            centroid[f] = km.centroids[c][f];

                            // Mandando o centroide.
                            MPI_Send(&centroid[0], N_FEATURES, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);

                        }
                    }
                }

                for(int p = 1; p < size; p++){
                    if(i+p-1 < km.n_instances){
                        // Recebendo do trabalhador p o rótulo do centroide mais perto da instância enviada anteriormente.
                        MPI_Recv(&label, 1, MPI_INT, p, 99, MPI_COMM_WORLD, &status);

                        // Atribuindo o rótulo no vetor de rótulos.
                        km.labels[i+p-1] = label;
                    }
                }
            }

            // Definindo a quantidade de instâncias por centroide.
            counter(&km);

            // Laço para atualizar os centroides.
            for(int c = 0; c < km.n_clusters; c += size-1){
                for(int p = 1; p < size; p++){

                    if(c+p-1 < km.n_clusters){
                        // Colocando o centroide num vetor auxiliar.
                        for (int f = 0; f < km.n_features; f++)
                            centroid[f] = km.centroids[c+p-1][f];

                        // Mandando o centroide.
                        MPI_Send(&centroid[0], N_FEATURES, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);

                        // Mandando quantas instâncias o centroide corrente tem.
                        MPI_Send(&km.cluster_count[c+p-1], 1, MPI_INT, p, 99, MPI_COMM_WORLD);

                        for (int i = 0; i < km.n_instances; i++) {

                            if (km.labels[i] == c+p-1) {

                                // Colocando a instância num vetor auxiliar.
                                for (int f = 0; f < km.n_features; f++)
                                    instance[f] = km.instances[i][f];

                                // Mandando a instância.
                                MPI_Send(&instance[0], km.n_features, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);

                            }
                        }
                    }
                }

                for(int p = 1; p < size; p++){

                    if(c+p-1 < km.n_clusters){

                        // Recebendo centroide atualizado.
                        MPI_Recv(&centroid[0], N_FEATURES, MPI_DOUBLE, p, 99, MPI_COMM_WORLD, &status);

                        // Recebendo o delta atual.
                        MPI_Recv(&current_delta, 1, MPI_DOUBLE, p, 99, MPI_COMM_WORLD, &status);

                        // Colocando a instância num vetor auxiliar.
                        for (int f = 0; f < km.n_features; f++)
                            km.centroids[c+p-1][f] = centroid[f];

                        mean_deltas += current_delta;
                    }
                }
                mean_deltas /= km.n_clusters;

            }

            for(int p = 1; p < size; p++){
                // Mandando para os processos a condição de parada.
                MPI_Send(&mean_deltas, 1, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);

        }
    }

    // Trabalhadores
    else{

            for(int i = rank-1; i < N_INSTANCES; i += size-1) {

                // Recebendo do mestre uma instância.
                MPI_Recv(&instance[0], N_FEATURES, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);

                label = nearest_centroid_id(instance, centroid);

                // Mandando para o mestre o rótulo do centroide mais perto da instância recebida.
                MPI_Send(&label, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
            }

            for(int c = rank-1; c < N_CLUSTERS; c += size-1) {

                // Zerando o vetor.
                for (int f = 0; f < N_FEATURES; f++)
                    new_centroid[f] = 0;

                current_delta = 0;

                // Recebendo do mestre um centroide.
                MPI_Recv(&centroid[0], N_FEATURES, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);

                // Recebendo do mestre o count do cluster corrente.
                MPI_Recv(&count, 1, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);

                for(int i = 0; i < count; i++){

                    // Recebendo do mestre cada instância.
                    MPI_Recv(&instance[0], N_FEATURES, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);

                    for (int f = 0; f < N_FEATURES; f++)
                        new_centroid[f] += instance[f]/count;
                }

                // Calculando deslocamento do centroide corrente.
                for (int f = 0; f < N_FEATURES; f++)
                    current_delta += pow(centroid[f] - new_centroid[f], 2);

                current_delta = sqrt(current_delta);

                // Mandando o novo centroide.
                MPI_Send(&new_centroid[0], N_FEATURES, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);

                // Mandando o deslocalmento.
                MPI_Send(&current_delta, 1, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);

            }

            // Recebendo do mestre a condição de parada.
            MPI_Recv(&mean_deltas, 1, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);

        }

        if (rank == 0)
            printf("Iteração: %d; Delta: %lf\n", iter, mean_deltas);

    } while(iter < MAX_ITER && mean_deltas > TOL);

     // obtém tempo e consumo de CPU depois da aplicação do filtro
	gettimeofday(&fim,0);
	getrusage(RUSAGE_SELF, &r2);

	printf("\nElapsed time:%f sec\tUser time:%f sec\tSystem time:%f sec\n",
	 (fim.tv_sec+fim.tv_usec/1000000.) - (inic.tv_sec+inic.tv_usec/1000000.),
	 (r2.ru_utime.tv_sec+r2.ru_utime.tv_usec/1000000.) - (r1.ru_utime.tv_sec+r1.ru_utime.tv_usec/1000000.),
	 (r2.ru_stime.tv_sec+r2.ru_stime.tv_usec/1000000.) - (r1.ru_stime.tv_sec+r1.ru_stime.tv_usec/1000000.));


    if(rank == 0){

        // Prints para a depuração
        // print_labels(&km);
        // print_centroids(&km);

        save_instances(&km);
        save_centroids(&km);
        save_labels(&km);

        free_k_means(&km);
    }

    MPI_Finalize();

    free(instance);
    free(centroid);
    free(new_centroid);
    instance = NULL;
    centroid = NULL;
    new_centroid = NULL;

    return 0;
}
