#ifndef _NODE_H
#define _NODE_H

// Types of nodes as codes
#define CT_LITERAL	1
#define CT_DATASET_VAR 2
#define CT_OPERATOR 3

#define OP_TIMES 42
#define OP_PLUS 43
#define OP_MINUS 45
#define OP_DIVIDE 47

struct node_t { 
    short c_type;
    struct node_t *left; 
    struct node_t *right;
    union content_u {
		int literal;
        int index_in_dataset;
		int operator_code;
	} content;
}; 

/*
    Creates a new node

    c_type can be CT_LITERAL, CT_DATASET_VAR or CT_OPERATOR
    if c_type == CT_LITERAL then -10 <= content <= 10
    if c_type == CT_DATASET_VAR then 0 <= content <= dataset length
    if c_type == CT_OPERATOR then content = OP_TIMES, OP_PLUS, OP_MINUS or OP_DIVIDE

*/
struct node_t* create_node(short c_type, int content);

/*
    Destroys the given node and all it's childs recursively
*/
void node_destroy(struct node_t *node);


#endif