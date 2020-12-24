#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

// Número de threads a serem utilizadas
#define N_THREADS 4

// Tipo de dado: k_means
typedef struct{
    int n_instances, n_features, n_clusters;
    double **instances, **centroids;
    int *labels;
} k_means;

// Tipo de dado: argumentos
typedef struct {
    k_means *k_m;
    int begin_offset, end_offset;
} arguments;

//------------------------------------------------------------------------------

void create_artificial_dataset(k_means *km){

    /*
        Função que cria um dataset artificial.
    */

    // Instanciando a matriz de instâncias.
    km->instances = (double **) malloc(km->n_instances*sizeof(double));

    for(int i = 0; i < km->n_instances; i++){

        // Alocando dinamicamente uma instância.
        km->instances[i] = (double *) malloc(km->n_features*sizeof(double));

        // Atribuindo valores as features.
        for (int f = 0; f < km->n_features; f++)
            km->instances[i][f] = i;//sqrt(i)*f*f;
    }

    // Alocando um vetor que contém os rótulos para cada istância.
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

//------------------------------------------------------------------------------

void select_centroids(k_means *km){

    /*
        Função que seleciona os centroides da primeira iteração.
    */

    // Instanciando a matriz de centroides.
    km->centroids = (double **) malloc(km->n_clusters*sizeof(double));

    for(int c = 0; c < km->n_clusters; c++){

        // Alocando dinamicamente um centroide.
        km->centroids[c] = (double *) malloc(km->n_features*sizeof(double));

        // Atribuindo valores para cada dimensão.
        for (int f = 0; f < km->n_features; f++)
            km->centroids[c][f] = km->instances[c][f];
    }
}

void print_centroids(k_means *km){

    /*
        Função que imprime os centroides.
    */

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
        Função que desaloca as variáveis dinâmicas.
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

void *nearest_centroid_id(arguments *arg){

    /*
        Função que calcula qual o índice do cluster mais próximo
        para cada instância dentro do intervalo dado pela struct arg.
    */

    int min_index;
    double current_dist, min_dist;

    printf("Begin and end: %d %d\n", arg->begin_offset, arg->end_offset);

    // Iterando entre as instâncias.
    for (int i = arg->begin_offset; i <= arg->end_offset; i++) {

        // Calculando a distância para cada cluster c.
        for (int c = 0; c < arg->k_m->n_clusters; c++){

            current_dist = 0;
            for (int f = 0; f < arg->k_m->n_features; f++)
                current_dist += pow((arg->k_m->centroids[c][f] - arg->k_m->instances[i][f]), 2);
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

        // Atribuindo o rótulo do cluster mais perto.
        arg->k_m->labels[i] = min_index;
    }
}

void label_instances(k_means *km){

    pthread_t threads[N_THREADS];
    arguments args[N_THREADS];
    int r;
    double a = km->n_instances / (float) N_THREADS;

    for (int t = 0; t < N_THREADS; t++) {

        args[t].k_m = km;
        args[t].begin_offset = ceil(a * t);
        args[t].end_offset = ceil(a * (t + 1)) - 1;

        // Criando as threads.
        r = pthread_create(&threads[t], NULL, &nearest_centroid_id, &args[t]);

        if (r != 0) {
            printf("Erro para criar uma thread (label_instances function)\n");
            exit(0);
        }

    }

    // Espera todas as threads terminarem sua execução.
    for(int t = 0; t < N_THREADS; t++)
        pthread_join(threads[t], NULL);

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

int main(int argc, char const *argv[]) {

    // Instanciando uma struct do tipo k_means e variáveis.
    k_means km;
    km.n_instances = 10;
    km.n_features = 3;
    km.n_clusters = 2;

    // Instanciando critérios de parada.
    double tol = 0.01, mean_deltas;
    int iter = 0;
    int max_iter = 10;

    create_artificial_dataset(&km);
    print_instances(&km);

    select_centroids(&km);
    print_centroids(&km);

    do {
        iter++;
        label_instances(&km);
        print_labels(&km);

        mean_deltas = update_centroids(&km);
        print_centroids(&km);
        printf("Iteração: %d; Delta: %lf\n", iter, mean_deltas);

    } while(iter < max_iter && mean_deltas > tol);

    free_k_means(&km);
    return 0;
}
