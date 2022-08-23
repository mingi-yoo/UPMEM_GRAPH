DPU_DIR := dpu
HOST_DIR := host
BUILDDIR ?= bin
NR_TASKLETS ?= 1
NR_DPUS ?= 1

HOST_TARGET := ${BUILDDIR}/host
BASE_TARGET := ${BUILDDIR}/pr_baseline
OURS_TARGET := ${BUILDDIR}/pr_ours


COMMON_INCLUDES := support
HOST_COMMON_INCLUDES := ${COMMON_INCLUDES}/support_host
DPU_COMMON_INCLUDES := ${COMMON_INCLUDES}/support_dpu

HOST_SOURCES := $(wildcard ${HOST_DIR}/*.cpp)
BASE_SOURCES := $(wildcard ${DPU_DIR}/baseline/*.c)
OURS_SOURCES := $(wildcard ${DPU_DIR}/ours/*.c)

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_INCLUDES}

HOST_FLAGS := g++ ${COMMON_FLAGS} -std=c++11 -O3 `dpu-pkg-config --cflags --libs dpu` -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS}
DPU_FLAGS := ${COMMON_FLAGS} -O2 -DNR_TASKLETS=${NR_TASKLETS}

all: ${HOST_TARGET} ${BASE_TARGET}

${HOST_TARGET}: ${HOST_SOURCES} ${HOST_COMMON_INCLUDES}
	$(CXX) -o $@ ${HOST_SOURCES} ${HOST_FLAGS}

${BASE_TARGET}: ${BASE_SOURCES} ${DPU_COMMON_INCLUDES}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${BASE_SOURCES}
}

clean:
	$(RM) -r $(BUILDDIR)

test: all
	./${HOST_TARGET} -i dataset/cora.txt -o ./