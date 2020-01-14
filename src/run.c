#include <stdio.h>
#include <stdlib.h>

#include "../include/tree_generator.h"
#include "../include/node.h"
#include "../include/stack.h"

#define NUM_TREES 1000

int main(int argc, char *argv[]) {
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
    push(stack, 2.0);
    push(stack, 2.0);
    push(stack, 2.0);
    push(stack, 2.0);
    printf("stack size %d", stack->size);
    pop(stack); pop(stack); pop(stack); pop(stack);
    printf("stack size %d", stack->size);
    free(stack);
}