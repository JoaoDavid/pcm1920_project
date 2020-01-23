#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <cuda.h>

extern "C" { 
    #include "../include/tree_generator.h"
    #include "../include/node.h"
    #include "../include/stack.h"
    #include "../include/dataset_parser.h"
}

#define NUM_TREES 10
#define NUM_GENERATIONS 3
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

__global__ void gpu_first_compute(double *dev_population, double *dev_target_values, int num_rows) {
    extern __shared__ double shared[];
    //populationULT(tree, row) population[tree * num_rows + row] 
    double res = pow(dev_population[blockIdx.x * num_rows + threadIdx.x] - dev_target_values[threadIdx.x], 2);
    //shared[threadIdx.x] = pow(dev_population[blockIdx.x * num_rows + threadIdx.x] - dev_target_values[threadIdx.x], 2);
    shared[threadIdx.x] = res;
    dev_population[blockIdx.x * num_rows + threadIdx.x] = res;
}

void gpu_prearation(double *population, double *target_values, int target_values_size, int population_size, int num_rows) {
    double *dev_population;
    double *dev_new_population;
    cudaMalloc(&dev_population, population_size);
    cudaMemcpy(dev_population, population, population_size, cudaMemcpyHostToDevice);
    cudaMalloc(&dev_new_population, population_size);
    int *res_tree_index = (int*) malloc((NUM_GENERATIONS+1)*sizeof(int));

    double *dev_target_values; //pointer to the location of the y's values in the gpu's memory
    cudaMalloc(&dev_target_values, target_values_size);
    cudaMemcpy(dev_target_values, target_values, target_values_size, cudaMemcpyHostToDevice);
    int *dev_fitness;
    int *dev_fitness_index;
    cudaMalloc(&dev_fitness, NUM_TREES);
    cudaMalloc(&dev_fitness_index, NUM_TREES);


    //Prints dataset content
    printf("---------- before PRINTING population CONTENT ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        for(int j = 0; j < num_rows; j++){
            printf("%f ", populationULT(i,j));
        }
        printf("\n");
    }

    gpu_first_compute<<<NUM_TREES, num_rows, sizeof(double) * num_rows>>>(dev_population, dev_target_values, num_rows);
    cudaMemcpy(population, dev_population, population_size, cudaMemcpyDeviceToHost);

    //Prints dataset content
    printf("---------- after PRINTING population CONTENT ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        for(int j = 0; j < num_rows; j++){
            printf("%f ", populationULT(i,j));
        }
        printf("\n");
    }
    //populationULT(tree, row) population[tree * num_rows + row]
}



//__global__ 
/*void gpu_compute(int curr_iteration, int num_rows, int num_trees) {
    

}*/


int main(int argc, char *argv[]) {
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

    gpu_prearation(population, target_values, target_values_size, population_size, num_rows);


    free(dataset);
    free(target_values);
    free(population);
    destroy_stack(stack);
    
    float average = (float)(total_size/NUM_TREES);
    printf("average tree size is %lf\n", average);
    for(int i = 0; i < NUM_TREES; i++) {
        node_destroy(trees[i]);
    }

}