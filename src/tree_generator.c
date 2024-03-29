#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<time.h> 

#include "../include/tree_generator.h"
#include "../include/node.h"

#define MAX_TREE_DEPTH 20
#define LITERAL_LOWER_BOUND -10
#define LITERAL_UPPER_BOUND 10


int nodes[3] = {CT_LITERAL, CT_DATASET_VAR, CT_OPERATOR};
int operators[4] = {OP_TIMES, OP_PLUS, OP_MINUS, OP_DIVIDE};

struct node_t* generate_tree(int num_vars) {
    struct node_t *root = create_node(CT_OPERATOR, operators[get_random(0,3)]);
    root->left = generate_tree_aux(num_vars-1, 0);
    root->right = generate_tree_aux(num_vars-1, 0);
    return root;
}

struct node_t* generate_tree_aux(int last_var_index, int curr_depth) {
    //switch(CT_OPERATOR){
    switch(nodes[get_random(0,2)]){
        case CT_LITERAL:{
            return create_node(CT_LITERAL, get_random(LITERAL_LOWER_BOUND,LITERAL_UPPER_BOUND));
            //break;
        }
        case CT_DATASET_VAR:{
            return create_node(CT_DATASET_VAR, get_random(0,last_var_index));
            //break;
        }
        case CT_OPERATOR:{
            struct node_t *node = create_node(CT_OPERATOR, operators[get_random(0,3)]);
            if (curr_depth < MAX_TREE_DEPTH) {
                node->left = generate_tree_aux(last_var_index, curr_depth++);
                node->right = generate_tree_aux(last_var_index, curr_depth++);
            }
            return node;
            //break;
        }
    }
    return NULL;
}

int tree_size(struct node_t* node) {   
  return node==NULL ? 0 : (tree_size(node->left) + 1 + tree_size(node->right));   
} 

/*
    Get random int between [lower,upper] (inclusive)
*/
int get_random(int lower, int upper) {
    return (rand() % (upper - lower + 1)) + lower;
}

void print_tree_rpn(struct node_t* node) {   
    if (node != NULL) {
        print_tree_rpn(node->left);
        print_tree_rpn(node->right);
        switch(node->c_type){
            case CT_LITERAL:{
                printf("%d ",node->content.literal);
                break;
            }
            case CT_DATASET_VAR:{
                printf("x%d ",node->content.index_in_dataset);          
                break;
            }
            case CT_OPERATOR:{
                switch(node->content.operator_code){
                    case OP_TIMES:{
                        printf("* "); 
                        break;
                    }
                    case OP_PLUS:{
                        printf("+ "); 
                        break;
                    }
                    case OP_MINUS:{
                        printf("- "); 
                        break;
                    }
                    case OP_DIVIDE:{
                        printf("/ "); 
                        break;
                    }
                }
                break;
            }
        }        
    }
} 

/*int main() {
    srand(time(0));
    struct node_t *trees[2];
    trees[0] = generate_tree();
    printf("\nsize %d\n", size(trees[0]));
    printf("literal %d\n", trees[0]->content.literal);
    printf("literal %d\n", trees[0]->content.index_in_dataset);
}*/