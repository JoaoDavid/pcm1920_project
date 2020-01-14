#ifndef _STACK_H
#define _STACK_H

struct stack_t { 
    int size;
    struct stack_node_t *head;
}; 

struct stack_node_t { 
    double value;
    struct stack_node_t *next;    
};

struct stack_t* create_stack();

void push(struct stack_t* stack, double value);

double pop(struct stack_t* stack);

double peek(struct stack_t* stack);

int isEmpty(struct stack_t* stack);

void toString(struct stack_t* stack);

void toStringAux(struct stack_node_t* node_stack);

void free_stack(struct stack_t* stack);

#endif