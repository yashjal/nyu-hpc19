CXX = g++ # define a variable CXX
CXXFLAGS = -std=c++11 -O3 -march=native

ifeq "$(CXX)" "icpc" # conditionals
CXXFLAGS += -qopenmp # for Intel
else
CXXFLAGS += -fopenmp # for GCC
endif

TARGETS = $(basename $(wildcard *.cpp)) $(basename $(wildcard *.c))

# default first rule
all : $(TARGETS)

# match all targets % such that there is a source file %.cpp
# Automatic variables: $< (first dependency), $^ (dependency list), $@ (target)
%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< -o $@

%:%.c *.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-$(RM) $(TARGETS)

.PHONY: all, clean

# References:
# https://www.gnu.org/software/make/manual/make.html
# http://web.mit.edu/gnu/doc/html/make_toc.html
