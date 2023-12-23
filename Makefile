CC=clang
SDL_INCLUDE_FLAGS=$(shell sdl2-config --cflags)
CFLAGS=-Wall -Wextra -Ofast $(SDL_INCLUDE_FLAGS)
LDLIBS=-lsdl2 -lpthread

default: build

run: build
	./particles

build: main.o simulation.o
	$(CC) $(CFLAGS) -o particles main.o simulation.o $(LDLIBS)

main.o: main.c simulation.h

simulation.o: simulation.c simulation.h