#include <stdio.h>
#include <stdlib.h>

#include "../include/tree_generator.h"
#include "../include/node.h"
#include "../include/stack.h"

#define NUM_TREES 1000
void processTree(const double *dataset, int tree_id, struct stack_t* stack, struct node_t* node);
void process_tree_aux(const double *dataset, int tree_id, struct stack_t* stack, struct node_t* node);

int num_entries = 1;
#define A(r, c) dataset[r * num_entries + c]

void processTree(const double *dataset, int tree_id, struct stack_t* stack, struct node_t* node) { 
    if (node == NULL) {
        return; 
    }        
  
    // first recur on left subtree 
    processTree(dataset, tree_id, stack, node->left); 
  
    // then recur on right subtree 
    processTree(dataset, tree_id, stack, node->right); 
  
    // now deal with the node 
    process_tree_aux(dataset, tree_id, stack, node);
}

void process_tree_aux(const double *dataset, int tree_id, struct stack_t* stack, struct node_t* node) {
    switch(node->c_type){
        case CT_LITERAL:{
            push(stack, (double)node->content.literal);
            //break;
        }
        case CT_DATASET_VAR:{
            //push(stack, dataset[tree_id][node->content.index_in_dataset]);
            push(stack, A(tree_id, node->content.index_in_dataset));            
            //break;
        }
        case CT_OPERATOR:{
            switch(node->content.operator_code){
                case OP_TIMES:{
                    double result = pop(stack) * pop(stack);
                    push(stack, result);
                    //break;
                }
                case OP_PLUS:{
                    double result = pop(stack) + pop(stack);
                    push(stack, result);
                    //break;
                }
                case OP_MINUS:{
                    double result = pop(stack) - pop(stack);
                    push(stack, result);
                    //break;
                }
                case OP_DIVIDE:{
                    double result = pop(stack) / pop(stack);
                    push(stack, result);
                    //break;
                }
            }

            //break;
        }
    }
}

int main(int argc, char *argv[]) {
    double dataset [2] = {2 ,4};
    /*struct node_t *trees[NUM_TREES];
    float total_size = 0;
    for(int i = 0; i < NUM_TREES; i++) {
        trees[i] = generate_tree();
        //printf("tree %d size: %d |", i, tree_size(trees[i]));
        total_size += tree_size(trees[i]);
    }
    float average = (float)(total_size/NUM_TREES);
    printf("average size is %lf", average);
    for(int i = 0; i < NUM_TREES; i++) {
        node_destroy(trees[i]);
    }*/

    struct stack_t* stack = create_stack();
    struct node_t *root = create_node(CT_OPERATOR, OP_PLUS);
    root->left = create_node(CT_LITERAL, 2);
    root->right = create_node(CT_OPERATOR, OP_PLUS);
    root->right->left = create_node(CT_LITERAL, 3);
    root->right->right = create_node(CT_LITERAL, 5);
    /* 4 becomes left child of 2 
           * 
         /   \ 
        2      + 
     /    \    /  \ 
    NULL NULL  3   5
    */
    processTree(dataset, 0, stack, root);
    double res = pop(stack);
    printf("resultado %lf", res);
    node_destroy(root);
    /*struct stack_t* stack = create_stack();
    push(stack, 3.9);
    push(stack, 1.4);
    push(stack, 2);
    push(stack, -10);
    printf("\n%lf", pop(stack));printf("\n%lf", pop(stack));printf("\n%lf", pop(stack));printf("\n%lf", pop(stack));*/

}