CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
LDLIBS = -lm

all: divide

divide: divide.c data.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

test: test.c divide.c data.h
	$(CC) $(CFLAGS) test.c -o $@ $(LDLIBS)

check: divide test
	./test
	./divide --steps 3 --tests 4 --context 32 --recent 4 --save smoke --csv smoke.csv
	./divide --load smoke.divide.weights --predict 'a3;b7;c1;d9;xyzxyz?a=' --recent 4
	./divide --load smoke.baseline.weights --predict 'a3;b7;c1;d9;xyzxyz?a=' --baseline
	rm -f smoke.divide.weights smoke.baseline.weights smoke.csv

clean:
	rm -f divide test

.PHONY: all check clean
