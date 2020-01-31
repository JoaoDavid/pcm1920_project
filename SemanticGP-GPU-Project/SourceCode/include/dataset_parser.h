#ifndef _DATASET_PARSER_H
#define _DATASET_PARSER_H

int parse_file_rows(char* filename);

int parse_file_columns(char* filename);

void parse_file_data(char* filename, float* dataset, float* target_values, int num_columns, int num_rows);
#endif