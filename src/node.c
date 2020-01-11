#include <stdio.h>
#include <stdlib.h>

#include "../include/node.h"
int not_main();
  
struct node_t* create_node(short c_type, int content) {
    // Allocate memory for new node
    struct node_t* node = (struct node_t*)malloc(sizeof(struct node_t)); 

    node->c_type = c_type;

    switch(c_type) {
        case CT_LITERAL:{
            node->content.literal = content;
            break;
        }
        case CT_DATASET_VAR:{
            node->content.index_in_dataset = content;
            break;
        }
        case CT_OPERATOR:{
            node->content.operator_code = content;
            break;
        }
    }
    // Initialize left and right children as NULL 
    node->left = NULL; 
    node->right = NULL; 
    return(node); 
}

void node_destroy(struct node_t *node) {
    if(node != NULL){
        node_destroy(node->left);
        node_destroy(node->right);
        free(node); 
    }
}

int not_main() 
{ 
  /*create root*/
  struct node_t *root = create_node(CT_OPERATOR, OP_TIMES);
  /* following is the tree after above statement  
  
        * 
      /   \ 
     NULL  NULL   
  */
    
  
  root->left = create_node(CT_LITERAL, 2);
  root->right = create_node(CT_OPERATOR, OP_PLUS);
  /* 2 and 3 become left and right children of 1 
           * 
         /   \ 
        2      + 
     /    \    /  \ 
    NULL NULL NULL NULL 
  */
  
  
  root->right->left = create_node(CT_LITERAL, 3);
  root->right->right = create_node(CT_LITERAL, 5);
  /* 4 becomes left child of 2 
           * 
         /   \ 
        2      + 
     /    \    /  \ 
    NULL NULL  3   5
*/
  

  node_destroy(root);
  return 0; 
}