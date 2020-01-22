#ifndef _TREE_GENERATOR_H
#define _TREE_GENERATOR_H

int get_random(int lower, int upper);
struct node_t* generate_tree(int num_vars);
struct node_t* generate_tree_aux(int num_vars, int curr_depth);
int tree_size(struct node_t* node);
void print_tree_rpn(struct node_t* node, char* result);

#endif