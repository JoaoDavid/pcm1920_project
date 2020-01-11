#ifndef _TREE_GENERATOR_H
#define _TREE_GENERATOR_H

int get_random(int lower, int upper);
struct node_t* generate_tree();
struct node_t* generate_tree_aux(int curr_depth);


#endif