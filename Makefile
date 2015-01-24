PROJECT = tdot-demo

CC = /usr/bin/g++-4.8
CFLAGS = -std=c++11
# CFLAGS for all configuration. cannot be overwritten by environment
override CFLAGS += -Wall -O3

RM = rm -f

C_INCL = ./include /usr/include
INCLUDES = $(addprefix -I, $(C_INCL))
C_LIB = opencv_core opencv_highgui opencv_video opencv_imgproc
LIBS = $(addprefix -l, $(C_LIB))

CPP_SRC  = main.cpp 					\
					 movingObject.cpp

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
	$(CC) -o $@ $(CFLAGS) $(INCLUDES) $(CPP_OBJS) $(LIBS)
	@echo 'Finished building $@'
	@echo ' '

schroot:
	schroot -c exp -- make $(PROJECT)

version:
	$(CC) --version

clean:
	$(RM) $(CPP_OBJS) $(DEPS) $(PROJECT)
