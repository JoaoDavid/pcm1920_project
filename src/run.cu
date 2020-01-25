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

#define NUM_TREES 32
#define NUM_GENERATIONS 2
/*#define NUM_TREES 4000
#define NUM_GENERATIONS 3000*/
void process_tree(const float *dataset, int num_vars, int row_index, struct stack_t* stack, struct node_t* node);
void process_tree_aux(const float *dataset, int num_vars, int row_index, struct stack_t* stack, struct node_t* node);


#define DATASET(row, column) dataset[row * num_vars + column]
#define populationULT(tree, row) population[tree * num_rows + row]

void process_tree(const float *dataset, int num_vars, int row_index, struct stack_t* stack, struct node_t* node) { 
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

void process_tree_aux(const float *dataset, int num_vars, int row_index, struct stack_t* stack, struct node_t* node) {
    switch(node->c_type){
        case CT_LITERAL:{
            //printf("%d ",node->content.literal);
            push(stack, (float)node->content.literal);
            break;
        }
        case CT_DATASET_VAR:{
            //push(stack, dataset[row_index][node->content.index_in_dataset]);
            float value = DATASET(row_index, node->content.index_in_dataset);
            //printf("value in dataset %f\n", value);
            //printf("%f ",value); 
            push(stack, value);            
            break;
        }
        case CT_OPERATOR:{
            switch(node->content.operator_code){
                case OP_TIMES:{
                    //printf("* ");
                    float result = pop(stack) * pop(stack);                    
                    push(stack, result);
                    break;
                }
                case OP_PLUS:{
                    //printf("+ ");
                    float result = pop(stack) + pop(stack);
                    push(stack, result);
                    break;
                }
                case OP_MINUS:{
                    //printf("- ");
                    float result = pop(stack) - pop(stack);
                    push(stack, result);
                    break;
                }
                case OP_DIVIDE:{
                    //printf("/ ");
                    float dividend = pop(stack);
                    float divisor = pop(stack);
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

__device__ float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_cpu(float x) {
    return 1 / (1 + exp(-x));
}


__global__ void gpu_generations(int *dev_matrix_gen, float *dev_old_fitness, float *dev_new_fitness) {
    //extern __shared__ float shared[];//shared_curr_matrix_line
    //shared[threadIdx.x] = dev_old_fitness[threadIdx.x];
    dev_matrix_gen[threadIdx.x] = threadIdx.x; //0 * NUM_TREES + threadIdx.x = threadIdx.x
    
    int min_fitness_index = 0;
    for(int gen = 1; gen < NUM_GENERATIONS; gen++) {
        //__syncthreads();
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
        //__syncthreads();
        dev_new_fitness[threadIdx.x] = dev_old_fitness[threadIdx.x] + sigmoid(dev_old_fitness[min_fitness_index]);
        //__syncthreads();
        dev_old_fitness[threadIdx.x] = dev_new_fitness[threadIdx.x];
    }
}



__global__ void gpu_generations_second(int gen, int *dev_matrix_gen, float *dev_old_fitness, float *dev_new_fitness, float *dev_fitness_aux, int *dev_fitness_index_aux) {
    extern __shared__ float shared[];
    int* index_fitness = (int*)&shared[0];
    float* values_fitness = (float*)&shared[blockDim.x];
    int index_fitness_global = blockDim.x * blockIdx.x + threadIdx.x; //certo
    int index_matrix_global = gen * gridDim.x * blockDim.x + index_fitness_global;//certo

    if (gen == 0) {
        dev_matrix_gen[index_matrix_global] = index_matrix_global;
    } else {
        if(gen % 2 == 0) {
            //int index_shared_mem = (threadIdx.x / 2) + ((blockDim.x/2) * (threadIdx.x%2));//certo
            /*index_fitness[index_shared_mem] = index_fitness_global;
            values_fitness[index_shared_mem] = dev_old_fitness[index_fitness_global];*/
            index_fitness[threadIdx.x] = index_fitness_global;
            values_fitness[threadIdx.x] = dev_old_fitness[index_fitness_global];
            __syncthreads();
            int i = blockDim.x / 2;
            while (i != 2) {
                if (threadIdx.x < i) {
                    if (values_fitness[threadIdx.x] > values_fitness[threadIdx.x + i]) {
                        values_fitness[threadIdx.x] = values_fitness[threadIdx.x + i];
                        index_fitness[threadIdx.x] = index_fitness[threadIdx.x + i];
                   }
                }
                __syncthreads();
                i /= 2;
            }
            __syncthreads();
            if (threadIdx.x < 2) {
                dev_matrix_gen[index_matrix_global] = index_fitness[threadIdx.x];
                dev_new_fitness[index_fitness_global] = values_fitness[threadIdx.x];
            }
            
        } else { ////////////////////////////////////
            index_fitness[threadIdx.x] = index_fitness_global;
            values_fitness[threadIdx.x] = dev_old_fitness[index_fitness_global];
            __syncthreads();
            int i = blockDim.x / 2;
            while (i != 0) {
                if (threadIdx.x < i) {
                    if (values_fitness[threadIdx.x] > values_fitness[threadIdx.x + i]) {
                        values_fitness[threadIdx.x] = values_fitness[threadIdx.x + i];
                        index_fitness[threadIdx.x] = index_fitness[threadIdx.x + i];
                   }
                }
                __syncthreads();
                i /= 2;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                dev_matrix_gen[index_matrix_global] = index_fitness[threadIdx.x];//remover mais tarde
                dev_new_fitness[index_fitness_global] = values_fitness[threadIdx.x];//remover mais tarde
                dev_fitness_aux[blockIdx.x] = values_fitness[threadIdx.x];
                dev_fitness_index_aux[blockIdx.x] = index_fitness[threadIdx.x];
            }
        }
        
    }
}

__global__ void gpu_generations_third(int gen, int *dev_matrix_gen, float *dev_old_fitness, float *dev_new_fitness, float *dev_fitness_aux, int *dev_fitness_index_aux) {
    extern __shared__ float shared[];
    int* index_fitness = (int*)&shared[0];
    float* values_fitness = (float*)&shared[blockDim.x];
    int index_fitness_global = blockDim.x * blockIdx.x + threadIdx.x; //certo
    int index_matrix_global = gen * gridDim.x * blockDim.x + index_fitness_global;//certo
    int min_fitness_index = 0;

    if (gen == 0) {
        dev_matrix_gen[index_matrix_global] = index_matrix_global;
    } else {
        int boundary = gridDim.x / 2;
        if (blockIdx.x < boundary) {
            min_fitness_index = 0;
            for(int i = min_fitness_index + 1; i < boundary; i++) {
                min_fitness_index = (dev_fitness_aux[min_fitness_index] < dev_fitness_aux[i] ? min_fitness_index : i);
            }
            dev_matrix_gen[index_matrix_global] = dev_fitness_index_aux[min_fitness_index];
        } else {
            min_fitness_index = boundary;
            for(int i = min_fitness_index + 1; i < gridDim.x; i++) {
                min_fitness_index = (dev_fitness_aux[min_fitness_index] < dev_fitness_aux[i] ? min_fitness_index : i);
            }
            dev_matrix_gen[index_matrix_global] = dev_fitness_index_aux[min_fitness_index];

        }
    }
}


__global__ void gpu_calc_init_fitness(float *dev_population, float *dev_target_values, float *dev_fitness) {
    extern __shared__ float shared[];
    shared[threadIdx.x] = pow(dev_population[blockIdx.x * blockDim.x + threadIdx.x] - dev_target_values[threadIdx.x], 2);
    int j = blockDim.x;
    int i = j/2;
    while (i != 0) {
        __syncthreads();
        if (threadIdx.x < i) {
            shared[threadIdx.x] += shared[threadIdx.x + i];
        }
        if (threadIdx.x == 0 && j % 2 != 0) {
            shared[threadIdx.x] += shared[i * 2];
        }
        j = i;
        i /= 2;
    }
    if(threadIdx.x == 0) {
        dev_fitness[blockIdx.x] = shared[threadIdx.x] / blockDim.x;
    }
}

void gpu_preparation(float *population, float *target_values, int *matrix_gen, float *gpu_fitness, int target_values_size, int population_size, int matrix_gen_size, int num_rows) {
    float *dev_population;
    cudaMalloc(&dev_population, population_size);
    cudaMemcpy(dev_population, population, population_size, cudaMemcpyHostToDevice);

    float *dev_target_values; //pointer to the location of the y's values in the gpu's memory
    cudaMalloc(&dev_target_values, target_values_size);
    cudaMemcpy(dev_target_values, target_values, target_values_size, cudaMemcpyHostToDevice);
    
    float *dev_fitness;
    cudaMalloc(&dev_fitness, NUM_TREES * sizeof(float));
    

    float *new_fitness = (float*) malloc(NUM_TREES * sizeof(float));
    float *dev_new_fitness;
    cudaMalloc(&dev_new_fitness, NUM_TREES * sizeof(float));


    //Prints dataset content
    /*printf("---------- before PRINTING population CONTENT ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        for(int j = 0; j < num_rows; j++){
            printf("%f ", populationULT(i,j));
        }
        printf("\n");
    }*/

    gpu_calc_init_fitness<<<NUM_TREES, num_rows, sizeof(float) * num_rows>>>(dev_population, dev_target_values, dev_fitness);
    cudaMemcpy(population, dev_population, population_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_fitness, dev_fitness, NUM_TREES*sizeof(float), cudaMemcpyDeviceToHost);

    //Prints dataset content
    /*printf("---------- after PRINTING population CONTENT ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        for(int j = 0; j < num_rows; j++){
            printf("%f ", populationULT(i,j));
        }
        printf("\n");
    }*/
    
    int num_threads_in_block = 8;

    //Prints fitness
    printf("---------- first fitness gpu ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        if(i % num_threads_in_block == 0){
            printf("\n");
        }
        printf("i:%d , %f | ",i, gpu_fitness[i]);
        
    }
    printf("\n");

    int *dev_matrix_gen;
    cudaMalloc(&dev_matrix_gen, matrix_gen_size);

    
    int num_blocks = NUM_TREES / num_threads_in_block;    
    float *dev_fitness_aux;
    cudaMalloc(&dev_fitness_aux, num_blocks * sizeof(float));
    int *dev_fitness_index_aux;
    cudaMalloc(&dev_fitness_index_aux, num_blocks * sizeof(int));

    int shared_memory_size = (sizeof(float) * num_threads_in_block) + (sizeof(int) * num_threads_in_block);

    gpu_generations_second<<<num_blocks,num_threads_in_block,shared_memory_size>>>(0, dev_matrix_gen, dev_fitness, dev_new_fitness, dev_fitness_aux, dev_fitness_index_aux);
    for(int gen = 1; gen < NUM_GENERATIONS; gen++){
        gpu_generations_second<<<num_blocks,num_threads_in_block,shared_memory_size>>>(gen, dev_matrix_gen, dev_fitness, dev_new_fitness, dev_fitness_aux, dev_fitness_index_aux);
        gpu_generations_third<<<num_blocks,num_threads_in_block,shared_memory_size>>>(gen, dev_matrix_gen, dev_fitness, dev_new_fitness, dev_fitness_aux, dev_fitness_index_aux);

    }

    
    //gpu_generations<<<1, NUM_TREES>>>(dev_matrix_gen, dev_fitness, dev_new_fitness);
    cudaMemcpy(matrix_gen, dev_matrix_gen, matrix_gen_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_fitness, dev_new_fitness, NUM_TREES*sizeof(float), cudaMemcpyDeviceToHost);

    printf("---------- PRINTING MATRIX GEN gpu ----------\n");
    for(int i = 0; i < NUM_GENERATIONS; i++) {
        for(int j = 0; j < NUM_TREES; j++){
            printf("%d ", matrix_gen[i * NUM_TREES + j]);
        }
        printf("\n");
    }

}



void cpu_seq_version(float *population, float *target_values, int *cpu_matrix_gen, float *old_fitness, int num_rows) {
    //float *old_fitness = (float*) malloc(NUM_TREES * sizeof(float));
    float *new_fitness = (float*) malloc(NUM_TREES * sizeof(float));
    //float *aux;

    for(int i = 0; i < NUM_TREES; i++) {
        float curr = 0;
        for(int j = 0; j < num_rows; j++){
            curr += pow(population[i * num_rows + j] - target_values[j],2);
        }
        old_fitness[i] = curr / num_rows;
    }
    printf("---------- first fitness cpu ----------\n");
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
        /*aux = old_fitness;
        old_fitness = new_fitness;
        new_fitness = aux;  */
        //printf("insnide gen\n");
        for(int i = 0; i < NUM_TREES; i++) {
            //printf("%f ", new_fitness[i]);
            old_fitness[i] = new_fitness[i];
        }//printf("\n");
    }
    //memcpy(fitness, new_fitness, NUM_TREES*sizeof(float));
    //memcpy(fitness, old_fitness, NUM_TREES);
    printf("---------- PRINTING MATRIX GEN  cpu----------\n");
    for(int i = 0; i < NUM_GENERATIONS; i++) {
        for(int j = 0; j < NUM_TREES; j++){
            printf("%d ", cpu_matrix_gen[i * NUM_TREES + j]);
        }
        printf("\n");
    }printf("\n");
}


int main(int argc, char *argv[]) {
    #define TIMER_START() gettimeofday(&tv1, NULL)
    #define TIMER_STOP()                                                           \
    gettimeofday(&tv2, NULL);                                                    \
    timersub(&tv2, &tv1, &tv);                                                   \
    time_delta = (float)tv.tv_sec + tv.tv_usec / 1000000.0

    struct timeval tv1, tv2, tv;
    float time_delta;


    //srand(time(NULL));
    //Parsing dataset file, and adding its values to the dataset array
    int num_columns = parse_file_columns(argv[1]); //x0,x1,x2,x3,...,xn and y
    int num_rows = parse_file_rows(argv[1]);
    int num_vars = num_columns - 1; //excluding y
    float* dataset = (float*) malloc((num_columns-1)*num_rows*sizeof(float));
    int target_values_size = num_rows*sizeof(float);
    float* target_values = (float*) malloc(target_values_size);
    parse_file_data(argv[1],dataset,target_values,num_columns,num_rows);
    printf("Dataset rows: %d\n",num_rows);
    printf("Dataset columns: %d\n",num_columns);    

    //Generating trees and processing the results with the dataset array
    struct node_t *trees[NUM_TREES];
    struct stack_t* stack = create_stack();
    float total_size = 0;
    int population_size = NUM_TREES*num_rows*sizeof(float);
    float* population = (float*) malloc(population_size);
    for(int i = 0; i < NUM_TREES; i++) {
        trees[i] = generate_tree(num_vars);
        //print_tree_rpn(trees[i]); printf("\n");
        total_size += tree_size(trees[i]);
        for(int j = 0; j < num_rows; j++){
            process_tree(dataset,num_vars,j,stack,trees[i]);
            populationULT(i,j) = pop(stack);
            //printf("Result (tree:%d|row:%d) %f\n", i, j, populationULT(i,j));
            clean_stack(stack);
        }
    }

    //Prints dataset content
    /*printf("---------- PRINTING DATASET CONTENT ----------\n");
    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_vars; j++){
            printf("%f ", DATASET(i,j));
        }
        printf("%f \n", target_values[i]);
    }*/

    int matrix_gen_size = NUM_TREES * NUM_GENERATIONS * sizeof(int);
    int *cpu_matrix_gen = (int*) malloc(matrix_gen_size);
    int *gpu_matrix_gen = (int*) malloc(matrix_gen_size);
    float *cpu_fitness = (float*) malloc(NUM_TREES * sizeof(float));
    float *gpu_fitness = (float*) malloc(NUM_TREES * sizeof(float));

    fprintf(stderr, "running on cpu...  ");
    TIMER_START();
    cpu_seq_version(population, target_values, cpu_matrix_gen, cpu_fitness, num_rows);
    TIMER_STOP();
    fprintf(stderr, "%f secs\n", time_delta);

    /*printf("---------- final fitness cpu ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        printf("%f , ", cpu_fitness[i]);
    }
    printf("\n");*/

    fprintf(stderr, "running on gpu...  ");
    TIMER_START();
    gpu_preparation(population, target_values, gpu_matrix_gen, gpu_fitness, target_values_size, population_size, matrix_gen_size, num_rows);
    TIMER_STOP();
    fprintf(stderr, "%f secs\n", time_delta);

    /*printf("---------- final fitness gpu ----------\n");
    for(int i = 0; i < NUM_TREES; i++) {
        printf("%f , ", gpu_fitness[i]);
    }
    printf("\n");*/

    free(dataset);
    free(target_values);
    free(population);
    destroy_stack(stack);
    
    float average = (float)(total_size/NUM_TREES);
    printf("average tree size is %lf\n", average);
    for(int i = 0; i < NUM_TREES; i++) {
        node_destroy(trees[i]);
    }    

    float espilon = 0.000001;
    for(int i = 0; i < NUM_TREES; i++) {
        if (cpu_fitness[i] - gpu_fitness[i] >= espilon) {
            printf("cpu and gpu final fitness arrays are different, values %f ; %f - FAIL!\n",cpu_fitness[i],gpu_fitness[i]);
            break;
        }        
    }
    if (memcmp(cpu_matrix_gen, gpu_matrix_gen, matrix_gen_size) != 0) {
        fprintf(stderr, "final matrix FAIL!- FAIL!- FAIL!- FAIL!\n");
    } else {
        printf("cpu and gpu matrixes are equal - OK\n");
    }

}