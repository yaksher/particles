CC=clang
SDL_INCLUDE_FLAGS=$(shell sdl2-config --cflags)
CFLAGS=-Wall -Wextra $(SDL_INCLUDE_FLAGS) -Ofast #-fsanitize=address -g
LDLIBS=-lsdl2 -lpthread

default: build

run: build
	./particles

force: clean build

build: main.o simulation.o
	$(CC) $(CFLAGS) -o particles main.o simulation.o $(LDLIBS)

main.o: main.c simulation.h

simulation.o: simulation.c simulation.h

clean:
	rm -f particles *.o