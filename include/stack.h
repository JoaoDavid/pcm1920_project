#ifndef _STACK_H
#define _STACK_H

struct stack_t { 
    int size;
    struct stack_node_t *head;
}; 

struct stack_node_t { 
    float value;
    struct stack_node_t *next;    
};

struct stack_t* create_stack();

void push(struct stack_t* stack, float value);

float pop(struct stack_t* stack);

float peek(struct stack_t* stack);

int isEmpty(struct stack_t* stack);

void toString(struct stack_t* stack);

void toStringAux(struct stack_node_t* node_stack);

void destroy_stack(struct stack_t* stack);

void clean_stack(struct stack_t* stack);

#endif