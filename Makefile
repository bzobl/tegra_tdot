PROJECT = tdot-demo

CC = g++
CFLAGS = -std=c++11
# CFLAGS for all configuration. cannot be overwritten by environment
override CFLAGS += -Wall -O3

RM = rm -f

C_INCL = ./include /opt/opencv3/include
INCLUDES = $(addprefix -I, $(C_INCL))

C_LIB_DIRS = /opt/opencv3/lib				\
						 /usr/local/cuda-6.0/lib
LIB_DIRS = $(addprefix -L, $(C_LIB_DIRS))

OPENCV_LIBS = core cuda cudaoptflow highgui imgproc objdetect imgcodecs videoio
C_LIB = $(addprefix opencv_, $(OPENCV_LIBS)) \
				pthread

LIBS = $(addprefix -l, $(C_LIB))

CPP_SRC  = main.cpp      					\
					 alpha-image.cpp				\
					 augmented-reality.cpp 	\
					 faces.cpp 							\
					 livestream.cpp   			\
					 optical-flow.cpp 			\
					 thread-safe-mat.cpp

CPP_H    = $(wildcard $(C_INCL)/*.h)

CPP_OBJS = $(CPP_SRC:%.cpp=%.o)
DEPS = $(CPP_OBJS:%.o=%.d) \
			 $(CUDA_OBJS:%.o=%.d)

all: $(PROJECT)

%.o: %.cpp $(CPP_H)
	@echo 'Building file: $<'
	$(CC) $(CFLAGS) $(INCLUDES) -c -o "$@" "$<"
	@echo 'Finished building $<'
	@echo ' '

$(PROJECT): $(CPP_OBJS)
	@echo 'Linking file: $@'
	$(CC) -o $@ $(CFLAGS) $(INCLUDES) $(LIB_DIRS) $(CPP_OBJS) $(LIBS)
	@echo 'Finished building $@'
	@echo ' '

schroot:
	schroot -c exp -- make $(PROJECT)

version:
	$(CC) --version

clean:
	$(RM) $(CPP_OBJS) $(DEPS) $(PROJECT)
