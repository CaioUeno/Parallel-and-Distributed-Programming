#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

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
    int *labels;
} k_means;

//------------------------------------------------------------------------------

void create_artificial_k_means(k_means *km){

    /*
        Função que cria um dataset artificial.
    */

    // Aloca a matriz de instâncias.
    km->instances = (double **) malloc(km->n_instances*sizeof(double));

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
    km->centroids = (double **) malloc(km->n_clusters*sizeof(double));

    // Aloca dinamicamente os centroides.
    for(int c = 0; c < km->n_clusters; c++){
        km->centroids[c] = (double *) malloc(km->n_features*sizeof(double));

        // Atribui valores às features.
        for (int f = 0; f < km->n_features; f++)
            km->centroids[c][f] = km->instances[c][f];
    }
}

int nearest_centroid_id(k_means *km, int i){

    /*
        Função que retorna o rótulo do centroide mais próximo da instância dada.
    */

    int min_index;
    double current_dist, min_dist;

    for (int c = 0; c < km->n_clusters; c++){
        current_dist = 0;

        // Calcula a distância euclidiana entre a instância e o centroide corrente.
        for (int f = 0; f < km->n_features; f++)
            current_dist += pow((km->centroids[c][f] - km->instances[i][f]), 2);
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

void label_instances(k_means *km){

    /*
        Função que atribui cada instância com o rótulo do centroide mais próximo.
    */

    for (int i = 0; i < km->n_instances; i++)
        km->labels[i] = nearest_centroid_id(km, i);
}

double update_centroids(k_means *km){

    /*
        Função que atualiza cada centroide e retorna a média dos deslocamentos.
    */

    int counter;
    double feature_sum, current_delta, mean_deltas = 0;

    for (int c = 0; c < km->n_clusters; c++){
        current_delta = 0;

        for (int f = 0; f < km->n_features; f++){
            counter = 0;
            feature_sum = 0;

            // Soma a feature corrente de todas as instâncias que pertencem ao centroide corrente e as conta.
            for (int i = 0; i < km->n_instances; i++)
                if(km->labels[i] == c){
                    counter++;
                    feature_sum += km->instances[i][f];
                }

            // Calcula o deslocamento dinamicamente (ao longo das iterações).
            current_delta += pow(km->centroids[c][f] - feature_sum/counter, 2);

            // Atualiza a feature (dimensão) corrente do centroide corrente.
            km->centroids[c][f] = feature_sum/counter;
        }

        // Finaliza o cálculo da distância euclidiana entre o centroide antigo e atualizado.
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
        Função que salva as instâncias.
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
        Função que salva os rótulos.
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

    // Rotula as instâncias e atualiza os centroides até satisfazer uma das condições (MAX_ITER ou TOL).
    do{
        iter++;

        label_instances(&km);
        mean_deltas = update_centroids(&km);

        printf("Iteração: %d; Delta: %lf\n", iter, mean_deltas);

    } while(iter < MAX_ITER && mean_deltas > TOL);

    // Obtém tempo e consumo de CPU após executar o algoritmo k-means.
	gettimeofday(&fim,0);
	getrusage(RUSAGE_SELF, &r2);

	printf("\nElapsed time:%f sec\tUser time:%f sec\tSystem time:%f sec\n",
	 (fim.tv_sec+fim.tv_usec/1000000.) - (inic.tv_sec+inic.tv_usec/1000000.),
	 (r2.ru_utime.tv_sec+r2.ru_utime.tv_usec/1000000.) - (r1.ru_utime.tv_sec+r1.ru_utime.tv_usec/1000000.),
	 (r2.ru_stime.tv_sec+r2.ru_stime.tv_usec/1000000.) - (r1.ru_stime.tv_sec+r1.ru_stime.tv_usec/1000000.));

    // Prints para a depuração
    // print_labels(&km);
    // print_centroids(&km);

    // Armazena os resultados em arquivo (.txt).
    save_instances(&km);
    save_centroids(&km);
    save_labels(&km);

    free_k_means(&km);

    return 0;
}
