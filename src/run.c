#include <stdio.h>
#include <stdlib.h>

#include "../include/tree_generator.h"
#include "../include/node.h"

#define NUM_TREES 1000

int main(int argc, char *argv[]) {
    struct node_t *trees[NUM_TREES];
    for(int i = 0; i < NUM_TREES; i++) {
        trees[i] = generate_tree();
    }
    for(int i = 0; i < NUM_TREES; i++) {
        node_destroy(trees[i]);
    }
}