#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/tree_generator.h"
#include "../include/node.h"
#include "../include/stack.h"
#include "../include/dataset_parser.h"

#define NUM_TREES 10
void process_tree(const double *dataset, int num_vars, int row_index, struct stack_t* stack, struct node_t* node);
void process_tree_aux(const double *dataset, int num_vars, int row_index, struct stack_t* stack, struct node_t* node);


#define DATASET(row, column) dataset[row * num_vars + column]
#define FUNCTION_RESULT(tree, row) function_res[tree * num_rows + row]

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


int main(int argc, char *argv[]) {
    //Parsing dataset file, and adding its values to the dataset array
    int num_columns = parse_file_columns(argv[1]); //x0,x1,x2,x3,...,xn and y
    int num_rows = parse_file_rows(argv[1]);
    int num_vars = num_columns - 1; //excluding y
    double* dataset = malloc((num_columns-1)*num_rows*sizeof(double));
    double* target_values = malloc(num_rows*sizeof(double));
    parse_file_data(argv[1],dataset,target_values,num_columns,num_rows);
    printf("Number of rows: %d\n",num_rows);
    printf("Number of columns: %d\n",num_columns);    

    struct node_t *trees[NUM_TREES];
    struct stack_t* stack = create_stack();
    float total_size = 0;    

    /*double* function_res = malloc(NUM_TREES*num_rows*sizeof(double));
    for(int i = 0; i < NUM_TREES; i++) {
        trees[i] = generate_tree(num_vars);
        print_tree_rpn(trees[i]); printf("\n");
        total_size += tree_size(trees[i]);
        for(int j = 0; j < num_rows; j++){
            process_tree(dataset,num_vars, j,stack,trees[i]);
            FUNCTION_RESULT(i,j) = pop(stack);
            printf("Result (tree:%d|row:%d) %f\n", i, j, FUNCTION_RESULT(i,j));
            clean_stack(stack);
        }
    }*/

    double* function_res = malloc(NUM_TREES*num_rows*sizeof(double));
    for(int i = 0; i < NUM_TREES; i++) {
        trees[i] = generate_tree(num_vars);
        print_tree_rpn(trees[i]); printf("\n");
        total_size += tree_size(trees[i]);
        for(int j = 0; j < num_rows; j++){
            process_tree(dataset,num_vars,j,stack,trees[i]);
            FUNCTION_RESULT(i,j) = pop(stack);
            printf("Result (tree:%d|row:%d) %f\n", i, j, FUNCTION_RESULT(i,j));
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

    free(dataset);
    free(target_values);
    destroy_stack(stack);
    
    float average = (float)(total_size/NUM_TREES);
    printf("average tree size is %lf\n", average);
    for(int i = 0; i < NUM_TREES; i++) {
        node_destroy(trees[i]);
    }

}