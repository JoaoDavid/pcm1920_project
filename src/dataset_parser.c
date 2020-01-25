#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/dataset_parser.h"

int parse_file_rows(char* filename){
    char c;
    int num_rows = 0;
    FILE* file = fopen(filename,"r");
    for (c = getc(file); c != EOF; c = getc(file)){
        if (c == '\n') // Increment count if this character is newline 
            num_rows++;
    }
    fclose(file);
    return num_rows;
}

int parse_file_columns(char* filename){
    FILE* file = fopen(filename,"r");
    int num_columns = 0;
    int bufferLength = 15000;
    char line[bufferLength];
    fseek(file, 0, SEEK_SET);
    fgets(line,bufferLength, file);
    char *p = strtok (line, "  ");
    while(p != NULL){
        p = strtok(NULL,"  ");
        num_columns++;
    }
    fclose(file);
    return num_columns;
}

void parse_file_data(char* filename, float* dataset, float* target_values, int num_columns, int num_rows){
    int bufferLength = 15000;
    char line[bufferLength];
    FILE* file = fopen(filename,"r");
    int row = 0, column = 0, test = 0;
    char *p = strtok (line, "  ");
    while (fgets(line, bufferLength, file) != NULL) {
        p = strtok (line, "  ");
        while(p != NULL){
            if(column == num_columns-1){
                target_values[row] = strtod(p, NULL);
                p = strtok(NULL,"  ");
                column++;
            }else{
                dataset[row * (num_columns-1) + column] = strtod(p, NULL);
                p = strtok(NULL,"  ");
                column++;
            }
        }
        column = 0;
        row++;
    }
}