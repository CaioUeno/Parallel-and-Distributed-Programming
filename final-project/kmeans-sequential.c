#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define a condição de parada do algoritmo
#define MAX_ITER 500
#define TOL 0.0001

// Define as quantidades de instâncias, características e grupos
#define N_INSTANCES 100
#define N_FEATURES 2
#define N_CLUSTERS 7

typedef struct{
    int n_instances, n_features, n_clusters;
    double **instances, **centroids;
    int *labels;
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
            km->instances[i][f] = sin(f+rand()%10)*cos(i+rand()%6)*(i+2)*pow(-1,rand()%2); //sqrt(i)*f*f;
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
}

int nearest_centroid_id(k_means *km, int i){

    int min_index;
    double current_dist, min_dist;

    for (int c = 0; c < km->n_clusters; c++){

        current_dist = 0;
        for (int f = 0; f < km->n_features; f++)
            current_dist += pow((km->centroids[c][f] - km->instances[i][f]), 2);
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

void label_instances_sequential(k_means *km){

    for (int i = 0; i < km->n_instances; i++)
        km->labels[i] = nearest_centroid_id(km, i); // Paralelizável \o/

}

double update_centroids(k_means *km){

    int counter;
    double aux, current_delta, mean_deltas = 0;

    for (int c = 0; c < km->n_clusters; c++) {
        current_delta = 0;
        for (int f = 0; f < km->n_features; f++){
            counter = 0;
            aux = 0;
            for (int i = 0; i < km->n_instances; i++){
                if(km->labels[i] == c){
                    counter++;
                    aux += km->instances[i][f];
                }
            }
            current_delta += pow(km->centroids[c][f] - aux/counter, 2);
            km->centroids[c][f] = aux/counter;
        }
        mean_deltas += sqrt(current_delta);
    }
    return mean_deltas/km->n_clusters;
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
    km->instances = NULL;
    km->centroids = NULL;
    km->labels = NULL;
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

    do {
        iter++;
        label_instances_sequential(&km);
        mean_deltas = update_centroids(&km);

        // Prints para a depuração
        // print_labels(&km);
        // print_centroids(&km);

        printf("Iteração: %d; Delta: %lf\n", iter, mean_deltas);

    } while(iter < MAX_ITER && mean_deltas > TOL);

    save_instances(&km);
    save_centroids(&km);
    save_labels(&km);

    free_k_means(&km);
    return 0;
}
