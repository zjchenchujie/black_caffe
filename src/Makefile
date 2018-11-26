#
# The following defines a variable named "NAME" with a value of "myprogram". By convention,
# a lowercase prefix (in this case "program") and an uppercased suffix (in this case "NAME"), separated
# by an underscore is used to name attributes for a common element. Think of this like
# using program.NAME, program.C_SRCS, etc. There are no structs in Make, so we use this convention
# to keep track of attributes that all belong to the same target or program.
#

PROJECT = caffeine

NAME := lib$(PROJECT).so
TEST_NAME := test_$(PROJECT)

CXX_SRCS := $(shell find caffeine ! -name "test_*.cpp" -name "*.cpp")
TEST_SRCS := $(shell find caffeine -name "test_*.cpp")
TEST_SRCS += gtest/gtest-all.cpp
PROTO_SRCS := $(wildcard caffeine/proto/*.proto)
PROTO_GEN_HEADER := $(PROTO_SRCS:.proto=.pb.h)
PROTO_GEN_CC := $(PROTO_SRCS:.proto=.pb.cc)
CXX_OBJS := $(CXX_SRCS:.cpp=.o)
PROTO_OBJS := $(PTORO_GEN_CC:.cc=.o)
OBJS := $(CXX_OBJS) $(PROTO_OBJS)
TEST_OBJS := $(TEST_SRCS:.cpp=.o)

CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE_DIR = $(CUDA_DIR)/include
CUDA_LIB_DIR = $(CUDA_DIR)/lib64

INCLUDE_DIRS := $(CUDA_INCLUDE_DIR) ./
LIBRARY_DIRS := $(CUDA_LIB_DIR)
LIBRARIES := cudart cublas
WARNINGS := -Wall

CXXFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir)) -fPIC
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(LIBRARIES),-l$(library))

#LINK = $(CXX) $(CXXFLAGS) $(LDFLAGS) $(WARNINGS)

.PHONY: all test clean distclean

all: $(NAME)

test: $(TEST_NAME)

$(NAME): $(PROTO_GEN_CC) $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(WARNING) $(OBJS) -shared -o $(NAME)

$(TEST_NAME):$(TEST_OBJS) $(OBJS)
	$(CXX) $(LDFALGS) $(WARNING) $(TEST_OBJS) $(OBJS) -o $(TEST_NAME)

$(PROTO_GEN_CC): $(PROTO_SRCS)
	protoc $(PROTO_SRCS) --cpp_out=.

clean:
	$(RM) $(NAME)
	$(RM) $(OBJS)
	$(RM) $(TEST_OBJS)
	$(RM) $(TEST_NAME)
	$(RM) $(PROTO_GEN_HEADER) $(PROTO_GEN_CC)


distclean: clean