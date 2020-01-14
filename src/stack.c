#include <stdio.h>
#include <stdlib.h>

#include "../include/stack.h"



struct stack_t* create_stack() {
    struct stack_t* stack = (struct stack_t*)malloc(sizeof(struct stack_t));
    stack->head = NULL;
    stack->size = 0;
    return stack;
}

void push(struct stack_t* stack, double value) {
    struct stack_node_t* stack_node = (struct stack_node_t*)malloc(sizeof(struct stack_node_t));
    stack_node -> next = stack->head;
    stack_node -> value = value;
    stack->head = stack_node;
    stack->size++;
}

double pop(struct stack_t* stack) {
    if (isEmpty(stack) != 1) {
        struct stack_node_t* stack_node = stack->head;
        stack->head = stack_node->next;
        stack->size--;
        double res = stack_node->value;
        //printf("value to be pooped %lf\n", res);
        free(stack_node);
        return res;
    } else {
        printf("entrou no pop vazio\n");
        return -1;
    }
}

double peek(struct stack_t* stack) {
    if (isEmpty(stack) != 1) {
        return stack->head->value;
    }
}

int isEmpty(struct stack_t* stack) {
    return stack->size==0 ? 1 : 0;
}

int size(struct stack_t* stack) {
    return stack->size;
}

void toString(struct stack_t* stack) {
    printf("HEAD: ");
    toStringAux(stack->head);
    printf("\n");
}

void toStringAux(struct stack_node_t* node_stack) {
    if (node_stack != NULL){
        printf("%f |", node_stack->value);
        toStringAux(node_stack->next);
    }
}

void destroy_stack(struct stack_t* stack){
    clean_stack(stack);
    free(stack);
}

void clean_stack(struct stack_t* stack){
    struct stack_node_t* node = stack->head;
    while(node != NULL){
        struct stack_node_t* temp = node->next;
        free(node);
        node = temp;
    }
}

