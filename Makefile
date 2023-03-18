CC = gcc -O2 -Wall -Wextra -Wpedantic -I"include/"

all: tests.exe

tests.exe: simd/f32.o tensor.o ops.o tests.o

# All linkage
%.exe:
	@echo " [LD]" $@
	@$(CC) $^ -o $@

# All .c files
%.o: %.c
	@echo " [CC]" $<
	@$(CC) -c -o $@ $<

clean:
	rm -f *.o *.exe
