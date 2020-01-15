#include <stdio.h>
#include <stdlib.h>

#include "../include/tree_generator.h"
#include "../include/node.h"
#include "../include/stack.h"

#define NUM_TREES 10
void process_tree(const double *dataset, int row_index, struct stack_t* stack, struct node_t* node);
void process_tree_aux(const double *dataset, int row_index, struct stack_t* stack, struct node_t* node);

int num_entries = 1;
#define A(r, c) dataset[r * num_entries + c]

void process_tree(const double *dataset, int row_index, struct stack_t* stack, struct node_t* node) { 
    if (node == NULL) {
        return; 
    }
    // then recur on right subtree 
    process_tree(dataset, row_index, stack, node->right);
    // first recur on left subtree 
    process_tree(dataset, row_index, stack, node->left);       
    // now deal with the node 
    process_tree_aux(dataset, row_index, stack, node);
}

void process_tree_aux(const double *dataset, int row_index, struct stack_t* stack, struct node_t* node) {
    switch(node->c_type){
        case CT_LITERAL:{
            //printf("%d ",node->content.literal);
            push(stack, (double)node->content.literal);
            break;
        }
        case CT_DATASET_VAR:{
            //push(stack, dataset[row_index][node->content.index_in_dataset]);
            double value = A(row_index, node->content.index_in_dataset);
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
                    double result = pop(stack) / pop(stack);
                    push(stack, result);
                    break;
                }
            }
            break;
        }
    }
}

int main(int argc, char *argv[]) {
    double dataset [2] = {2 ,4};
    struct node_t *trees[NUM_TREES];
    struct stack_t* stack = create_stack();
    float total_size = 0;    
    int num_columns = 1; //x0,x1,x2,x3,...,xn and y
    int num_rows = 1;
    double function_res[NUM_TREES][num_rows];
    for(int i = 0; i < NUM_TREES; i++) {
        trees[i] = generate_tree(num_columns);
        print_tree(trees[i]); printf("\n");
        total_size += tree_size(trees[i]);
        for(int j = 0; j < num_rows; j++){
            process_tree(dataset,j,stack,trees[i]);
            function_res[i][j] = pop(stack);
            printf("Result %f\n", function_res[i][j]);
            clean_stack(stack);
        }
    }

    
    
    destroy_stack(stack);

    float average = (float)(total_size/NUM_TREES);
    printf("average size is %lf", average);
    for(int i = 0; i < NUM_TREES; i++) {
        node_destroy(trees[i]);
    }
    /*
    struct stack_t* stack = create_stack();
    struct node_t *root = create_node(CT_OPERATOR, OP_TIMES);
    root->left = create_node(CT_LITERAL, 1);
    root->right = create_node(CT_OPERATOR, OP_PLUS);
    root->right->left = create_node(CT_LITERAL, 3);
    root->right->right = create_node(CT_LITERAL, 5);
    node_destroy(root);
    destroy_stack(stack);*/
    /* 4 becomes left child of 2 
           - 
         /   \ 
        10      + 
     /    \    /  \ 
    NULL NULL  3   5
    */
   /*
    processTree(dataset, 0, stack, root);
    double res = pop(stack);
    printf("resultado %f ", res);
    node_destroy(root);*/
    /*struct stack_t* stack = create_stack();
    push(stack, 3.9);
    push(stack, 1.4);
    push(stack, 2);
    push(stack, -10);
    toString(stack);
    printf("\n%lf", pop(stack));printf("\n%lf", pop(stack));printf("\n%lf", pop(stack));printf("\n%lf", pop(stack));*/

}