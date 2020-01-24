#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

extern "C" { 
    #include "../include/tree_generator.h"
    #include "../include/node.h"
    #include "../include/stack.h"
    #include "../include/dataset_parser.h"
}

#define NUM_TREES 5
#define NUM_GENERATIONS 4
void process_tree(const double *dataset, int num_vars, int row_index, struct stack_t* stack, struct node_t* node);
void process_tree_aux(const double *dataset, int num_vars, int row_index, struct stack_t* stack, struct node_t* node);


#define DATASET(row, column) dataset[row * num_vars + column]
#define populationULT(tree, row) population[tree * num_rows + row]

void process_tree(const double *dataset, int num_vars, int row_index, struct stack_t* stack, struct node_t* node) { 
    if (node == NULL) {
        return; 
    }
    // then recur on right subtree 
    process_tree(dataset, num_vars, row_index, stack, node->right);
    // first recur on left subtree 
    process_tree(dataset, num_vars, row_index, stack, node->left);       
    // now deal with the node 
    process_tree_aux(dataset, num_vars, row_index, stack, node);
}

void process_tree_aux(const double *dataset, int num_vars, int row_index, struct stack_t* stack, struct node_t* node) {
    switch(node->c_type){
        case CT_LITERAL:{
            //printf("%d ",node->content.literal);
            push(stack, (double)node->content.literal);
            break;
        }
        case CT_DATASET_VAR:{
            //push(stack, dataset[row_index][node->content.index_in_dataset]);
            double value = DATASET(row_index, node->content.index_in_dataset);
            //printf("value in dataset %f\n", value);
            //printf("%f ",value); 
            push(stack, value);            
            break;
        }
        case CT_OPERATOR:{
            switch(node->content.operator_code){
                case OP_TIMES:{
                    //printf("* ");
                    double result = pop(stack) * pop(stack);                    
                    push(stack, result);
                    break;
                }
                case OP_PLUS:{
                    //printf("+ ");
                    double result = pop(stack) + pop(stack);
                    push(stack, result);
                    break;
                }
                case OP_MINUS:{
                    //printf("- ");
                    double result = pop(stack) - pop(stack);
                    push(stack, result);
                    break;
                }
                case OP_DIVIDE:{
                    //printf("/ ");
                    double dividend = pop(stack);
                    double divisor = pop(stack);
                    if (divisor == 0) {
                        push(stack, 0);
                    } else {
                        push(stack, dividend/divisor);
                    }                    
                    break;
                }
            }
            break;
        }
    }
}

__device__ double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_cpu(double x) {
    return 1 / (1 + exp(-x));
}


__global__ void gpu_generations(int *dev_matrix_gen, double *dev_old_fitness, double *dev_new_fitness) {
    extern __shared__ double shared[];//shared_curr_matrix_line
    shared[threadIdx.x] = dev_old_fitness[threadIdx.x];
    dev_matrix_gen[threadIdx.x] = threadIdx.x; //0 * NUM_TREES + threadIdx.x = threadIdx.x
    
    int min_fitness_index = 0;
    for(int gen = 1; gen < NUM_GENERATIONS; gen++) {
        __syncthreads();
        if (gen % 2 == 0) {
            min_fitness_index = threadIdx.x % 2;
            for(int i = min_fitness_index + 2; i < NUM_TREES; i += 2) {
                min_fitness_index = (dev_old_fitness[min_fitness_index] < dev_old_fitness[i] ? min_fitness_index : i);
            }
            dev_matrix_gen[gen * NUM_TREES + threadIdx.x] = min_fitness_index;
        } else {
            int boundary = NUM_TREES / 2;
            if (threadIdx.x < boundary) {
                min_fitness_index = 0;
                for(int i = min_fitness_index + 1; i < boundary; i++) {
                    min_fitness_index = (dev_old_fitness[min_fitness_index] < dev_old_fitness[i] ? min_fitness_index : i);
                }
                dev_matrix_gen[gen * NUM_TREES + threadIdx.x] = min_fitness_index;
            } else {
                min_fitness_index = boundary;
                for(int i = min_fitness_index + 1; i < NUM_TREES; i++) {
                    min_fitness_index = (dev_old_fitness[min_fitness_index] < dev_old_fitness[i] ? min_fitness_index : i);
                }
                dev_matrix_gen[gen * NUM_TREES + threadIdx.x] = min_fitness_index;
            }
        }
        //Calculate new fitness
        __syncthreads();
        dev_new_fitness[threadIdx.x] = dev_old_fitness[threadIdx.x] + sigmoid(dev_old_fitness[min_fitness_index]);
        __syncthreads();
        dev_old_fitness[threadIdx.x] = dev_new_fitness[threadIdx.x];
    }
}


__global__ void gpu_first_fitness(double *dev_population, double *dev_target_values, int num_rows, double *dev_fitness) {
    extern __shared__ double shared[];
    //populationULT(tree, row) population[tree * num_rows + row] 
    double res = pow(dev_population[blockIdx.x * num_rows + threadIdx.x] - dev_target_values[threadIdx.x], 2);
    //shared[threadIdx.x] = pow(dev_population[blockIdx.x * num_rows + threadIdx.x] - dev_target_values[threadIdx.x], 2);
    shared[threadIdx.x] = res;
    dev_population[blockIdx.x * num_rows + threadIdx.x] = res;
    __syncthreads();
    int i = num_rows/2;
    while (i != 0) {
        if (threadIdx.x < i) {
            shared[threadIdx.x] += shared[threadIdx.x + i];
        }            
        __syncthreads();
        i /= 2;
    }

    if(threadIdx.x == 0) {
        dev_fitness[blockIdx.x] = shared[threadIdx.x] / num_rows;
    }
}

void gpu_prearation(double *population, double *target_values, int *matrix_gen, int target_values_size, int population_size, int matrix_gen_size, int num_rows) {
    double *dev_population;
    cudaMalloc(&dev_population, population_size);
    cudaMemcpy(dev_population, population, population_size, cudaMemcpyHostToDevice);

    double *dev_target_values; //pointer to the location of the y's values in the gpu's memory
    cudaMalloc(&dev_target_values, target_values_size);
    cudaMemcpy(dev_target_values, target_values, target_values_size, cudaMemcpyHostToDevice);
    
    double *fitness = (double*) malloc(NUM_TREES * sizeof(double));
    double *dev_fitness;
    cudaMalloc(&dev_fitness, NUM_TREES * sizeof(double));

    double *new_fitness = (double*) malloc(NUM_TREES * sizeof(double));
    double *dev_new_fitness;
    cudaMalloc(&dev_new_fitness, NUM_TREES * sizeof(double));


    //Prints dataset content
    printf("---------- before PRINTING population CONTENT ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        for(int j = 0; j < num_rows; j++){
            printf("%f ", populationULT(i,j));
        }
        printf("\n");
    }

    gpu_first_fitness<<<NUM_TREES, num_rows, sizeof(double) * num_rows>>>(dev_population, dev_target_values, num_rows, dev_fitness);
    cudaMemcpy(population, dev_population, population_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fitness, dev_fitness, NUM_TREES*sizeof(double), cudaMemcpyDeviceToHost);

    //Prints dataset content
    printf("---------- after PRINTING population CONTENT ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        for(int j = 0; j < num_rows; j++){
            printf("%f ", populationULT(i,j));
        }
        printf("\n");
    }

    //Prints fitness
    printf("---------- PRINTING fitness ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        printf("%f , ", fitness[i]);
    }
    printf("\n");
    int *dev_matrix_gen;
    cudaMalloc(&dev_matrix_gen, matrix_gen_size);
    gpu_generations<<<1, NUM_TREES, sizeof(double) * NUM_TREES * 2>>>(dev_matrix_gen, dev_fitness, dev_new_fitness);
    cudaMemcpy(matrix_gen, dev_matrix_gen, matrix_gen_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_fitness, dev_new_fitness, NUM_TREES*sizeof(double), cudaMemcpyDeviceToHost);

    printf("---------- PRINTING MATRIX GEN ----------\n");
    for(int i = 0; i < NUM_GENERATIONS; i++) {
        for(int j = 0; j < NUM_TREES; j++){
            printf("%d ", matrix_gen[i * NUM_TREES + j]);
        }
        printf("\n");
    }

    printf("---------- PRINTING new fitness ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        printf("%f , ", new_fitness[i]);
    }
    printf("\n");

}



void cpu_seq_version(double *population, double *target_values, int *cpu_matrix_gen, int num_rows) {
    double *old_fitness = (double*) malloc(NUM_TREES * sizeof(double));
    double *new_fitness = (double*) malloc(NUM_TREES * sizeof(double));
    double *aux;

    for(int i = 0; i < NUM_TREES; i++) {
        double curr = 0;
        for(int j = 0; j < num_rows; j++){
            curr += pow(population[i * num_rows + j] - target_values[j],2);
        }
        old_fitness[i] = curr / num_rows;
    }
    printf("---------- PRINTING fitness cpu ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        printf("%f , ", old_fitness[i]);
    }
    printf("\n");
    // second part of the algorithm
    for(int i = 0; i < NUM_TREES; i++) { //gen = 0
        cpu_matrix_gen[i] = i;
    }
    for(int gen = 1; gen < NUM_GENERATIONS; gen++) {
        if (gen % 2 == 0) {
            int min_fitness_index_even = 0;
            int min_fitness_index_odd = 1;
            for(int i = 2; i < NUM_TREES; i++) {
                if (i % 2 == 0) {
                    min_fitness_index_even = (old_fitness[min_fitness_index_even] < old_fitness[i] ? min_fitness_index_even : i);
                } else {
                    min_fitness_index_odd = (old_fitness[min_fitness_index_odd] < old_fitness[i] ? min_fitness_index_odd : i);
                }
            }
            for(int i = 0; i < NUM_TREES; i++) {
                if (i % 2 == 0) {
                    cpu_matrix_gen[gen * NUM_TREES + i] = min_fitness_index_even;
                    new_fitness[i] = old_fitness[i] + sigmoid_cpu(old_fitness[min_fitness_index_even]);
                } else {
                    cpu_matrix_gen[gen * NUM_TREES + i] = min_fitness_index_odd;
                    new_fitness[i] = old_fitness[i] + sigmoid_cpu(old_fitness[min_fitness_index_odd]);
                }
            }
        } else {
            int boundary = NUM_TREES / 2;
            int min_fitness_index_first_half = 0;
            int min_fitness_index_second_half = boundary;
            for(int i = 1; i < NUM_TREES; i++) {
                if (i < boundary) {
                    min_fitness_index_first_half = (old_fitness[min_fitness_index_first_half] < old_fitness[i] ? min_fitness_index_first_half : i);
                } else {
                    min_fitness_index_second_half = (old_fitness[min_fitness_index_second_half] < old_fitness[i] ? min_fitness_index_second_half : i);
                }
            }
            for(int i = 0; i < NUM_TREES; i++) {
                if (i < boundary) {
                    cpu_matrix_gen[gen * NUM_TREES + i] = min_fitness_index_first_half;
                    new_fitness[i] = old_fitness[i] + sigmoid_cpu(old_fitness[min_fitness_index_first_half]);
                } else {
                    cpu_matrix_gen[gen * NUM_TREES + i] = min_fitness_index_second_half;
                    new_fitness[i] = old_fitness[i] + sigmoid_cpu(old_fitness[min_fitness_index_second_half]);
                }
            }
        }
        aux = old_fitness;
        old_fitness = new_fitness;
        new_fitness = aux;
    }
    old_fitness = new_fitness;
    printf("---------- PRINTING MATRIX GEN  cpu----------\n");
    for(int i = 0; i < NUM_GENERATIONS; i++) {
        for(int j = 0; j < NUM_TREES; j++){
            printf("%d ", cpu_matrix_gen[i * NUM_TREES + j]);
        }
        printf("\n");
    }printf("\n"); 
}


int main(int argc, char *argv[]) {
    //srand(time(NULL));
    //Parsing dataset file, and adding its values to the dataset array
    int num_columns = parse_file_columns(argv[1]); //x0,x1,x2,x3,...,xn and y
    int num_rows = parse_file_rows(argv[1]);
    int num_vars = num_columns - 1; //excluding y
    double* dataset = (double*) malloc((num_columns-1)*num_rows*sizeof(double));
    int target_values_size = num_rows*sizeof(double);
    double* target_values = (double*) malloc(target_values_size);
    parse_file_data(argv[1],dataset,target_values,num_columns,num_rows);
    printf("Number of rows: %d\n",num_rows);
    printf("Number of columns: %d\n",num_columns);    

    struct node_t *trees[NUM_TREES];
    struct stack_t* stack = create_stack();
    float total_size = 0;
    int population_size = NUM_TREES*num_rows*sizeof(double);
    double* population = (double*) malloc(population_size);
    for(int i = 0; i < NUM_TREES; i++) {
        trees[i] = generate_tree(num_vars);
        print_tree_rpn(trees[i]); printf("\n");
        total_size += tree_size(trees[i]);
        for(int j = 0; j < num_rows; j++){
            process_tree(dataset,num_vars,j,stack,trees[i]);
            populationULT(i,j) = pop(stack);
            printf("Result (tree:%d|row:%d) %f\n", i, j, populationULT(i,j));
            clean_stack(stack);
        }
    }

    //Prints dataset content
    printf("---------- PRINTING DATASET CONTENT ----------\n");
    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_vars; j++){
            printf("%f ", DATASET(i,j));
        }
        printf("%f \n", target_values[i]);
    }

    int matrix_gen_size = NUM_TREES * NUM_GENERATIONS * sizeof(int);
    int *cpu_matrix_gen = (int*) malloc(matrix_gen_size);
    int *gpu_matrix_gen = (int*) malloc(matrix_gen_size);

    cpu_seq_version(population, target_values, cpu_matrix_gen, num_rows);

    gpu_prearation(population, target_values, gpu_matrix_gen, target_values_size, population_size, matrix_gen_size, num_rows);


    free(dataset);
    free(target_values);
    free(population);
    destroy_stack(stack);
    
    float average = (float)(total_size/NUM_TREES);
    printf("average tree size is %lf\n", average);
    for(int i = 0; i < NUM_TREES; i++) {
        node_destroy(trees[i]);
    }

    if (memcmp(cpu_matrix_gen, gpu_matrix_gen, matrix_gen_size) != 0) {
        fprintf(stderr, "MATRIX GEN CPU DIFFERENT FROM GPU - FAIL!\n");
    } else {
        printf("MATRIX GEN CPU EQUAL TO GPU - OK\n");
    }

    /*if (memcmp(cpu_matrix_gen, gpu_matrix_gen, matrix_gen_size) != 0) {
        fprintf(stderr, "FITNESS FAIL!\n");
    } else {
        printf("FITNESS OK\n");
    }*/

}