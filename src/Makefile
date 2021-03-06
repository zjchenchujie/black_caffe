#
# The following defines a variable named "NAME" with a value of "myprogram". By convention,
# a lowercase prefix (in this case "program") and an uppercased suffix (in this case "NAME"), separated
# by an underscore is used to name attributes for a common element. Think of this like
# using program.NAME, program.C_SRCS, etc. There are no structs in Make, so we use this convention
# to keep track of attributes that all belong to the same target or program.
#
PROJECT := caffe
NAME := lib$(PROJECT).so
TEST_NAME := test_$(PROJECT)
CXX_SRCS := $(shell find caffe ! -name "test_*.cpp" -name "*.cpp")
CUDA_SRCS := $(shell find caffe -name "*.cu")
TEST_SRCS := $(shell find caffe/test -name "test_*.cpp")
GTEST_SRCS := gtest/gtest-all.cpp
PROTO_SRCS := $(wildcard caffe/proto/*.proto)
PROTO_GEN_HEADER := ${PROTO_SRCS:.proto=.pb.h}
PROTO_GEN_CC := ${PROTO_SRCS:.proto=.pb.cc}
PROTO_GEN_PY := ${PROTO_SRCS:.proto=_pb2.py}
CXX_OBJS := ${CXX_SRCS:.cpp=.o}
CUDA_OBJS := ${CUDA_SRCS:.cu=.cuo}
PROTO_OBJS := ${PROTO_SRCS:.proto=.pb.o}
OBJS := $(PROTO_OBJS) $(CXX_OBJS) $(CUDA_OBJS)
TEST_OBJS := ${TEST_SRCS:.cpp=.o}
GTEST_OBJS := ${GTEST_SRCS:.cpp=.o}
TEST_BINS := ${TEST_OBJS:.o=.testbin}

CUDA_DIR := /usr/local/cuda
CUDA_ARCH := -arch=sm_30
MKL_DIR := /opt/intel/mkl

CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
MKL_INCLUDE_DIR := $(MKL_DIR)/include
MKL_LIB_DIR := $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64

INCLUDE_DIRS := . $(CUDA_INCLUDE_DIR) $(MKL_INCLUDE_DIR)
LIBRARY_DIRS := . $(CUDA_LIB_DIR) $(MKL_LIB_DIR)
LIBRARIES := cuda cudart cublas protobuf pthread glog mkl_rt mkl_intel_thread curand leveldb snappy
WARNINGS := -Wall

CXXFLAGS +=   -fPIC $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(LIBRARIES),-l$(library))

LINK = $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(WARNINGS)
NVCC = nvcc ${CXXFLAGS:-fPIC=-Xcompiler -fPIC} $(CPPFLAGS) $(CUDA_ARCH)

.PHONY: all test clean distclean linecount

all: $(NAME)

test: $(OBJS) $(GTEST_OBJS) $(TEST_BINS)

runtest: test
	for testbin in $(TEST_BINS); do $$testbin; done

$(TEST_BINS): %.testbin: %.o
	$(CXX) $< $(OBJS) $(GTEST_OBJS) -o $@ $(LDFLAGS) $(WARNINGS)

linecount: clean
	cloc --read-lang-def=caffe.cloc caffe/

#$(TEST_NAME): $(OBJS) $(TEST_OBJS)
#	$(CXX) $(TEST_OBJS) $(OBJS) -o $(TEST_NAME) $(LDFLAGS) $(WARNINGS)
#	./$(TEST_NAME)

$(NAME): $(PROTO_GEN_CC) $(OBJS)
	$(LINK) -shared $(OBJS) -o $(NAME)

$(CUDA_OBJS): %.cuo: %.cu
	$(NVCC) -c $< -o $@

$(PROTO_GEN_CC): $(PROTO_SRCS)
	protoc $(PROTO_SRCS) --cpp_out=. --python_out=.

clean:
	$(RM) $(NAME) $(TEST_BINS)
	$(RM) $(OBJS) $(TEST_OBJS)
	$(RM) $(PROTO_GEN_HEADER) $(PROTO_GEN_CC) $(PROTO_GEN_PY)

distclean: clean
