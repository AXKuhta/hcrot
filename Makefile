CC = gcc -O2 -Wall -Wextra -Wpedantic

all: tests.exe

tests.exe: tensor.o

# All linkage
%.exe:
	@echo " [LD]" $<
	@$(CC) $^ -o $@

# All .c files
%.o: %.c
	@echo " [CC]" $<
	@$(CC) -c -o $@ $<

clean:
	rm -f *.o *.exe
