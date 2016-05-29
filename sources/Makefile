### Ce makefile compile les sources de la librairie et envoie la librairie statique (extension .a) dans ../lib ###

# Specify compiler
CC = g++

# Specify extensions of files to delete when cleaning
CLEANEXTS = o a 

# Specify the target file, the install directory and the include directory
OUTPUTFILE  = libeasySD.a
INSTALLDIR  = ../lib
INCDIR = ../inc

# Specify the compiler options
CFLAGS = -std=c++11 -O2 -Wall -Wextra

# Specify object files
OBJECT_FILES = test.o loaders.o vectorizers.o naive_bayes.o utils.o

# Specify source files
SOURCE_FILES = test.cpp loaders.cpp vectorizers.cpp naive_bayes.cpp utils.cpp

# Specify header files
HEADER_FILES = test.hpp loaders.hpp vectorizers.hpp naive_bayes.hpp utils.hpp

# Default target
.PHONY: all
all: $(OUTPUTFILE)

# Build libtest.a from test.o
$(OUTPUTFILE): $(OBJECT_FILES)
	ar ru $@ $^
	ranlib $@

$(OBJECT_FILES): $(SOURCE_FILES) $(HEADER_FILES)
	$(CC) $(CFLAGS) -c $(SOURCE_FILES)

.PHONY: install
install:$(OUTPUTFILE)
	mkdir -p $(INSTALLDIR)
	cp -p $(OUTPUTFILE) $(INSTALLDIR)
	mkdir -p $(INCDIR)
	cp -p *.h *.hpp $(INCDIR)

.PHONY: clean 
clean:
	for file in $(CLEANEXTS); do rm -f *.$$file; done

# Indicate dependencies of .ccp files on .hpp files
# test.o: test.cpp test.hpp