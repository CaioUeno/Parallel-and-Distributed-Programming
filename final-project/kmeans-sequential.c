#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct{
    int n_instances, n_features, n_clusters;
    double **instances, **centroids;
    int *labels;
} dataset;

//------------------------------------------------------------------------------

void create_artificial_dataset(dataset *ds){

    /*
        Função que cria um dataset artificial.
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

void print_instances(dataset *ds){

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

void select_centroids(dataset *ds){

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

void print_centroids(dataset *ds){

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

void print_labels(dataset *ds){

    /*
        Função que imprime os rótulos.
    */

    printf("Rótulos: \n");
    for (int i = 0; i < ds->n_instances; i++)
        printf("%d \n", ds->labels[i]);

    printf("\n");
}


//------------------------------------------------------------------------------

void free_dataset(dataset *ds){

    /*
        Função que desaloca as dataset.
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

int nearest_centroid_id(dataset *ds, int i){

    int min_index;
    double current_dist, min_dist;

    for (int c = 0; c < ds->n_clusters; c++){

        current_dist = 0;
        for (int f = 0; f < ds->n_features; f++)
            current_dist += pow((ds->centroids[c][f] - ds->instances[i][f]), 2);
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

void label_instances_sequential(dataset *ds){

    for (int i = 0; i < ds->n_instances; i++)
        ds->labels[i] = nearest_centroid_id(ds, i); // Paralelizável \o/

}

double update_centroids(dataset *ds){

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

    dataset ds;
    ds.n_instances = 10;
    ds.n_features = 3;
    ds.n_clusters = 2;

    double tol = 0.0001, mean_deltas;
    int iter = 0;
    int max_iter = 10;

    create_artificial_dataset(&ds);
    print_instances(&ds);

    select_centroids(&ds);
    print_centroids(&ds);

    do {
        iter++;
        label_instances_sequential(&ds);
        print_labels(&ds);

        mean_deltas = update_centroids(&ds);
        print_centroids(&ds);
        printf("Delta: %lf\n", mean_deltas);

    } while(iter < max_iter && mean_deltas > tol);

    free_dataset(&ds);
    return 0;
}
