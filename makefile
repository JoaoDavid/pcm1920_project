CC = gcc
LB = lb

BIN_dir = binary
OBJ_dir = object
LIB_dir = lib
SRC_dir = src
INCLUDE_dir = include

CFLAGS = -Wall -g -I 

#-----------------------

#Header Files
node.o = node.h


all: client-lib.o table-client table-server

table-client : $(OBJECTS_table_client)
	$(CC) -pthread $(LIB_dir)/client-lib.o  $(OBJ_dir)/table-client.o -o $(BIN_dir)/$@
	
table-server : $(OBJECTS_table_server) 
	$(CC) -pthread $(addprefix $(OBJ_dir)/,$^) -o $(BIN_dir)/$@

	
client-lib.o: $(C_client_lib)
	$(LD) -r $(addprefix $(OBJ_dir)/,$^) -o $(LIB_dir)/$@

%.o: $(SRC_dir)/%.c $($@)
	$(CC) $(CFLAGS) $(INCLUDE_dir) -o $(OBJ_dir)/$@ -c $<
	
clean:
	rm -f $(OBJ_dir)/*
	rm -f $(BIN_dir)/*
	rm -f $(LIB_dir)/*

runTestValgrindTableClient:
	valgrind --leak-check=full $(BIN_dir)/table-client

runTestValgrindTableServer:
	valgrind --leak-check=full $(BIN_dir)/table-server


