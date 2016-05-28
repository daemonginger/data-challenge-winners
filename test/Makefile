### Ce makefile permet d'utiliser la lib précompilée à partir d'un fichier .cpp ###
# Changer LIB_PATH si le nom de la lib change. A noter que g++ transforme automatiquement le nom de la lib donnée. Par exemple, si on lui dit test, il ira cherhcher libtest.a

# Specify the compiler
CC = g++

# Specify compiler options
CFLAGS = -std=c++11 -O2 -Wall -Wextra
INC_PATH = -I ../inc
LIB_PATH = -L ../lib -l easySD

# Specify the target file
OUTPUTFILE  = main

# Specify the object files
OBJECT_FILES = main.o

# Specify source files
SOURCE_FILES = main.cpp

# Default target
.PHONY: all
all: $(OUTPUTFILE)

# Build target from object files
$(OUTPUTFILE): $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(INC_PATH) $(OBJECT_FILES) $(LIB_PATH) -o $(OUTPUTFILE)

# Build object files from sources
$(OBJECT_FILES): $(SOURCE_FILES)
	$(CC) $(CFLAGS) $(INC_PATH) $(LIB_PATH) -c $(SOURCE_FILES)

.PHONY: clean 
clean:
	rm -f main.o main

.PHONY: run
run: $(OUTPUTFILE)
	./$(OUTPUTFILE)