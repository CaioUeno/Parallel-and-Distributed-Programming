#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

// Define a condição de parada do algoritmo
#define MAX_ITER 1
#define TOL 0.0001

// Define as quantidades de instâncias, características e grupos
#define N_INSTANCES 4
#define N_FEATURES 2
#define N_CLUSTERS 2

typedef struct{
    int n_instances, n_features, n_clusters;
    double **instances, **centroids;
    int *labels;
} k_means;


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
            km->instances[i][f] = i; //sin(f+rand()%10)*cos(i+rand()%6)*(i+2)*pow(-1,rand()%2);
    }

    km->labels = (int *) malloc(km->n_instances*sizeof(int));

}

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
    km->instances = NULL;
    km->centroids = NULL;
    km->labels = NULL;
}

int nearest_centroid_id(double *inst, double *cent){

    int min_index;
    double current_dist, min_dist;
    MPI_Status status;

    for (int c = 0; c < N_CLUSTERS; c++){

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
//
// void label_instances_sequential(k_means *km){
//
//     for (int i = 0; i < km->n_instances; i++)
//         km->labels[i] = nearest_centroid_id(km, i); // Paralelizável \o/
//
// }









int main(int argc, char **argv) {

    int rank, size, n_process, type;
    MPI_Status status;

    // Inicializa MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double mean_deltas = 0;
    double *instance, *centroid;
    int iter = 0;

    instance = (double *) malloc(N_FEATURES*sizeof(double));
    centroid = (double *) malloc(N_FEATURES*sizeof(double));
    int label;

    do {
        iter++;

        if (rank == 0) {

            k_means km;
            km.n_instances = N_INSTANCES;
            km.n_features = N_FEATURES;
            km.n_clusters = N_CLUSTERS;

            create_artificial_k_means(&km);
            select_centroids(&km);


            for(int i = 0; i < km.n_instances; i += size-1){
                for(int p = 1; p < size; p++){

                    if(i+p-1 < km.n_instances){


                      for (int f = 0; f < km.n_features; f++)
                        instance[f] = km.instances[i+p-1][f];


                      // Manda uma instância
                      MPI_Send(&instance[0], km.n_features, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);
                      // printf("Mestre mandou para trabalhador %d instância %d.\n", p, i+p-1);


                      for (int c = 0; c < km.n_clusters; c++) {

                          for (int f = 0; f < km.n_features; f++)
                            centroid[f] = km.centroids[c][f];

                          MPI_Send(&centroid[0], N_FEATURES, MPI_DOUBLE, p, 99, MPI_COMM_WORLD);
                          // printf("Mestre mandou para trabalhador %d centroide %d.\n", p, c);
                      }

                      // Recebendo do trabalhador o rótulo
                      MPI_Recv(&label, 1, MPI_INT, p, 99, MPI_COMM_WORLD, &status);
                      // printf("Mestre recebeu do trabalhador %d rótulo da instância %d.\n", p, i+p-1);

                      km.labels[i+p-1] = label;
                    }
                }
            }


            save_instances(&km);
            save_centroids(&km);
            save_labels(&km);

            free_k_means(&km);
        }

        else{


            for(int i = rank-1; i < N_INSTANCES; i = i+size-1) {

                // Recebendo do mestre a instância
                MPI_Recv(&instance[0], N_FEATURES, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);
                // printf("Trabalhador %d recebeu do mestre instância %d.\n", rank, i);

                label = nearest_centroid_id(instance, centroid);
                MPI_Send(&label, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
                // printf("Trabalhador %d mandou o rótulo da instância %d.\n", rank, i);
            }

        }


        // mean_deltas = update_centroids(&km);

        // Prints para a depuração
        // print_labels(&km);
        // print_centroids(&km);

        // printf("Iteração: %d; Delta: %lf\n", iter, mean_deltas);

    } while(iter < MAX_ITER && mean_deltas > TOL);

    MPI_Finalize();

    free(instance);
    free(centroid);
    instance = NULL;
    centroid = NULL;

    return 0;
}
