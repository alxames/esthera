MTGP_SRC = mtgp/mtgp32-fast.c \
	   mtgp/mtgp32-opencl.c \
	   mtgp/mtgp32dc-param-11213.c

MTGP_TEX_SRC = mtgp/mtgp32-fast.c \
	       mtgp/mtgp32-opencl-tex.c \
	       mtgp/mtgp32dc-param-11213.c

SRC = pf_generic.c
KERNEL = pf_generic.cl bitonic-sort.cl reduce.cl resampling-valias.cl
MODEL_PATH = ../../models/robotarm/opencl
ROBOTARM = $(MODEL_PATH)/pf_model.h $(MODEL_PATH)/pf_model.cl

CC = g++
INC = -I/usr/local/cuda/include -I/opt/pkg/AMD-APP-SDK-v2.6-RC3-lnx64/include
LIB = -lrt -L/usr/local/cuda/lib64 -L/opt/pkg/AMD-APP-SDK-v2.6-RC3-lnx64/lib/x86_64 -lOpenCL

MTGP_FLAG   = -Imtgp -DPRNG_MTGP

DEBUG = -DGLOBAL_ONLY -DTIMING_HOST# -DDETERMINISTIC -DDEBUG_ESTIMATE -DDEBUG -g #-DTIMING_DEVICE -G
OPTI  = -O3# -DVALIAS_RESAMPLING -DADAPTIVE_RESAMPLING
WARN  = #-Wall -Wextra

NVCC_OPTIONS = $(DEBUG) $(OPTI) $(ARCH) --compiler-options '$(WARN)'

.PHONY: clean

all: pf_robotarm_mtgp pf_robotarm_mtgp_tex

pf_robotarm_mtgp: ${SRC} ${KERNEL} ${ROBOTARM} ${MTGP_SRC}
	${CC} ${INC} ${DEBUG} ${OPTI} ${WARN} -o $@ ${SRC} ${LIB} ${MTGP_SRC} ${MTGP_FLAG} -I$(MODEL_PATH)

pf_robotarm_mtgp_tex: ${SRC} ${KERNEL} ${ROBOTARM} ${MTGP_TEX_SRC}
	${CC} ${INC} ${DEBUG} ${OPTI} ${WARN} -o $@ ${SRC} ${LIB} ${MTGP_TEX_SRC} ${MTGP_FLAG} -DPRNG_MTGP_TEX -I$(MODEL_PATH)

test-valias: test-valias.cpp
	${CC} ${INC} ${DEBUG} ${WARN} -o $@ $< ${LIB}

run: pf_robotarm_mtgp
	./pf_robotarm_mtgp -v -c -m 64 -N 16 -t 1 -l 1 $(MODEL_PATH)/../input5

clean:
	rm -f *.o pf_robotarm_mtgp pf_robotarm_mtgp_tex test-valias

