SRC = pf_generic.c

ROBOTARM_SRC = robotarm/pf_model.c
EDRO_SRC = edro/pf_model.c

SFMT_SRC = SFMT-1.3.3/SFMT.c
SFMT_INC = -I SFMT-1.3.3/ -DPRNG_SFMT -DMEXP=11213

CC = gcc
INC = -I libsimdmath
LIB = -lrt -lm
CFLAGS = -static -Wall -Wextra -std=gnu99 -O3 -msse3 -fno-strict-aliasing -DHAVE_SSE2 -flax-vector-conversions -DVALIAS_RESAMPLING -DGLOBAL_ONLY -DTIMING_HOST #-DDEBUG_ESTIMATE #-DDEBUG -g

.PHONY: clean

all: pf_robotarm

pf_robotarm: ${SRC} ${ROBOTARM_SRC} ${SFMT_SRC}
	${CC} ${CFLAGS} ${INC} ${SFMT_INC} -o $@ ${SRC} ${ROBOTARM_SRC} ${SFMT_SRC} -Irobotarm ${LIB}

pf_edro: ${SRC} ${EDRO_SRC} ${SFMT_SRC}
	${CC} ${CFLAGS} ${INC} ${SFMT_INC} -o $@ ${SRC} ${EDRO_SRC} ${SFMT_SRC} -Iedro ${LIB}

clean:
	rm -f *.o pf_robotarm pf_edro

