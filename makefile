CC = gcc
CFLAGS = -I.

BIN_dir = binary
OBJ_dir = object
LIB_dir = lib
SRC_dir = src
INCLUDE_dir = include

#-----------------------

all: run

run: $(OBJ_dir)/run.o $(OBJ_dir)/node.o $(OBJ_dir)/stack.o $(OBJ_dir)/tree_generator.o
	$(CC) -o $(BIN_dir)/run $(OBJ_dir)/run.o $(OBJ_dir)/node.o $(OBJ_dir)/stack.o $(OBJ_dir)/tree_generator.o

run.o: $(SRC_dir)/run.c
	$(CC) -c $(SRC_dir)/run.c -o $(OBJ_dir)/run.o

node.o: $(SRC_dir)/node.c
	$(CC) -c $(SRC_dir)/node.c -o $(OBJ_dir)/node.o

stack.o: $(SRC_dir)/stack.c
	$(CC) -c $(SRC_dir)/stack.c -o $(OBJ_dir)/stack.o

tree_generator.o: $(SRC_dir)/tree_generator.c
	$(CC) -c $(SRC_dir)/tree_generator.c -o $(OBJ_dir)/tree_generator.o
	
clean:
	rm -f $(OBJ_dir)/*
	rm -f $(BIN_dir)/*
	rm -f $(LIB_dir)/*

valgrindRun:
	valgrind --leak-check=full $(BIN_dir)/run

