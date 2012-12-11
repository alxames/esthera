MTGP_SOURCES = mtgp/mtgp32-cuda.cu \
               mtgp/mtgp32-cuda-common.cu \
               mtgp/mtgp-cuda-common.cu \
               mtgp/mtgp32-fast.c \
               mtgp/mtgp32dc-param-11213.c

MTGP_HEADERS = mtgp/mtgp32-cuda.h \
               mtgp/mtgp32-cuda-common.h \
               mtgp/mtgp-cuda-common.h \
               mtgp/mtgp32-fast.h \
               mtgp/mtgp32dc-param-11213.h

HOST := $(shell hostname)

NVCC = nvcc
ARCH =
INC =
LIB =

CUDALINK = -lcuda

DEBUG = -DGLOBAL_ONLY -DDEBUG_ESTIMATE -O3 #-DTIMING_HOST -DTIMING_DEVICE -DDEBUG -g -G
OPTI  = #-O0 --use_fast_math
WARN  = -Wall -Wextra -Wno-unused-parameter -Wno-unused-function

NVCC_OPTIONS = $(DEBUG) $(OPTI) $(ARCH) --compiler-options '$(WARN)'

MODEL_PATH = ../../models/robotarm/cuda

.PHONY:    clean

all: pf_robotarm pf_robotarm_soa pf_robotarm_soa_packed

pf_cuda: $(MTGP_SOURCES) $(MTGP_HEADERS) Makefile
	${NVCC} ${NVCC_OPTIONS} ${INC} ${LIB} ${CUDALINK} -o $@ particle_filter.cu $(MTGP_SOURCES)

pf_robotarm: pf_generic.cu $(MODEL_PATH)/pf_model.cuh Makefile
	${NVCC} ${NVCC_OPTIONS} ${INC} ${LIB} ${CUDALINK} -o $@ pf_generic.cu -I$(MODEL_ARM) -lcurand

pf_robotarm_soa: pf_generic_soa.cu $(MODEL_PATH)/pf_model_soa.cuh Makefile
	${NVCC} ${NVCC_OPTIONS} ${INC} ${LIB} ${CUDALINK} -o $@ pf_generic_soa.cu -I$(MODEL_ARM) -lcurand

pf_robotarm_soa_packed: pf_generic_soa_packed.cu $(MODEL_PATH)/pf_model_soa_packed.cuh Makefile
	${NVCC} ${NVCC_OPTIONS} ${INC} ${LIB} ${CUDALINK} -o $@ pf_generic_soa_packed.cu -I$(MODEL_ARM) -lcurand

clean:
	rm -f *.o pf_cuda pf_robotarm pf_robotarm_soa pf_robotarm_soa_packed

