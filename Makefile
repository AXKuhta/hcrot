CC = gcc -O2 -ftree-vectorize -march=sandybridge -Wall -Wextra -Wpedantic -Wvla -I"include/"

all: tests.exe bench.exe

lib.a: simd/f32.o tensor.o print.o index.o ops.o
tests.exe: tests.o lib.a
bench.exe: bench.o lib.a

%.a: 
	@echo " [AR]" $@
	@ar rcs $@ $^

# All linkage
%.exe:
	@echo " [LD]" $@
	@$(CC) $^ -o $@

# All .c files
%.o: %.c
	@echo " [CC]" $<
	@$(CC) -c -o $@ $<

clean:
	rm -f *.o *.a *.exe
