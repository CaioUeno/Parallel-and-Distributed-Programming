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

void create_artificial_k_means(k_means *ds){

    /*
        Função que cria um k_means artificial.
    */

    // Instanciando a matriz de instâncias.
    ds->instances = (double **) malloc(ds->n_instances*sizeof(double));

    for(int i = 0; i < ds->n_instances; i++){

        // Alocando dinamicamente uma instância
        ds->instances[i] = (double *) malloc(ds->n_features*sizeof(double));

        // Atribuindo valores as features
        for (int f = 0; f < ds->n_features; f++)
            ds->instances[i][f] = i; //sqrt(i)*f*f;
    }

    ds->labels = (int *) malloc(ds->n_instances*sizeof(int));

}

void print_instances(k_means *ds){

    /*
        Função que imprime as instâncias.
    */

    printf("Instâncias: \n");
    for (int i = 0; i < ds->n_instances; i++) {
        for (int f = 0; f < ds->n_features; f++)
            printf("%lf ", ds->instances[i][f]);
        printf("\n");
    }
    printf("\n");
}

//------------------------------------------------------------------------------

void select_centroids(k_means *ds){

    /*
        Função que seleciona os centroides da primeira iteração.
    */

    // Instanciando a matriz de centroides.
    ds->centroids = (double **) malloc(ds->n_clusters*sizeof(double));

    for(int c = 0; c < ds->n_clusters; c++){

        // Alocando dinamicamente um centroide
        ds->centroids[c] = (double *) malloc(ds->n_features*sizeof(double));

        // Atribuindo valores as features
        for (int f = 0; f < ds->n_features; f++)
            ds->centroids[c][f] = ds->instances[c][f];
    }
}

void print_centroids(k_means *ds){

    /*
        Função que imprime os centroides.
    */

    if (ds->n_clusters == 0) {
        printf("É necessário definir os clusters antes de printá-los!\n");
        exit(0);
    }

    printf("Centroides: \n");
    for (int c = 0; c < ds->n_clusters; c++) {
        for (int f = 0; f < ds->n_features; f++)
            printf("%lf ", ds->centroids[c][f]);
        printf("\n");
    }
    printf("\n");
}

void print_labels(k_means *ds){

    /*
        Função que imprime os rótulos.
    */

    printf("Rótulos: \n");
    for (int i = 0; i < ds->n_instances; i++)
        printf("%d \n", ds->labels[i]);

    printf("\n");
}


//------------------------------------------------------------------------------

void free_k_means(k_means *ds){

    /*
        Função que desaloca as k_means.
    */

    for(int i = 0; i < ds->n_instances; i++){
        free(ds->instances[i]);
        ds->instances[i] = NULL;
    }

    for(int i = 0; i < ds->n_clusters; i++){
        free(ds->centroids[i]);
        ds->centroids[i] = NULL;
    }

    free(ds->instances);
    free(ds->centroids);
    free(ds->labels);
    ds->instances = NULL;
    ds->centroids = NULL;
    ds->labels = NULL;
}

//------------------------------------------------------------------------------

void *nearest_centroid_id(arguments *arg){

    int min_index;
    double current_dist, min_dist;

    printf("Begin and end: %d %d\n", arg->begin_offset, arg->end_offset);

    for (int i = arg->begin_offset; i <= arg->end_offset; i++) {

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

        arg->k_m->labels[i] = min_index;
    }
}

void label_instances(k_means *ds){

    pthread_t threads[N_THREADS];
    arguments args[N_THREADS];
    int r;
    double a = ds->n_instances / (float) N_THREADS;

    for (int t = 0; t < N_THREADS; t++) {

        args[t].k_m = ds;
        args[t].begin_offset = ceil(a * t);
        args[t].end_offset = ceil(a * (t + 1)) - 1;

        // Criando as threads
        r = pthread_create(&threads[t], NULL, nearest_centroid_id, &args[t]);

        if (r != 0) {
            printf("Erro para criar thread (label instances)\n");
            exit(0);
        }

    }

    // Espera todas as threads terminarem sua execução
    for(int t = 0; t < N_THREADS; t++)
        pthread_join(threads[t], NULL);

}

double update_centroids(k_means *ds){

    int counter;
    double aux, current_delta, mean_deltas = 0;

    for (int c = 0; c < ds->n_clusters; c++) {
        current_delta = 0;
        for (int f = 0; f < ds->n_features; f++){
            counter = 0;
            aux = 0;
            for (int i = 0; i < ds->n_instances; i++){
                if(ds->labels[i] == c){
                    counter++;
                    aux += ds->instances[i][f];
                }
            }
            current_delta += pow(ds->centroids[c][f] - aux/counter, 2);
            ds->centroids[c][f] = aux/counter;
        }
        mean_deltas += sqrt(current_delta);
    }
    return mean_deltas/ds->n_clusters;
}

//------------------------------------------------------------------------------

int main(int argc, char const *argv[]) {

    k_means ds;
    ds.n_instances = 10;
    ds.n_features = 3;
    ds.n_clusters = 2;

    double tol = 0.0001, mean_deltas;
    int iter = 0;
    int max_iter = 10;

    create_artificial_k_means(&ds);
    print_instances(&ds);

    select_centroids(&ds);
    print_centroids(&ds);

    do {
        iter++;
        label_instances(&ds);
        print_labels(&ds);

        mean_deltas = update_centroids(&ds);
        print_centroids(&ds);
        printf("Delta: %lf\n", mean_deltas);

    } while(iter < max_iter && mean_deltas > tol);

    free_k_means(&ds);
    return 0;
}
