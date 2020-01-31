#!/bin/bash
nvcc src/node.c src/tree_generator.c src/stack.c src/dataset_parser.c src/run.cu -o binary/run