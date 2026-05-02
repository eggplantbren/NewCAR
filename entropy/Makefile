# Put the directory above 'DNest4' into an environment variable
# called DNEST4_PATH for this to work.

CC = g++
CXXFLAGS = -std=c++20 -O3 -march=native -Wall -Wextra -pedantic -DNDEBUG
LIBS = -ldnest4 -lpthread
INCLUDE = -I PostEnt2026 -I . -I $(DNEST4_PATH) -I PostEnt2026/pybind11/include -I /usr/include/python3.12
LIBS = -L $(DNEST4_PATH)/DNest4/code

default:
	$(CXX) -Wno-unused-variable -Wno-attributes -Wno-unused-parameter -DNO_THREADS $(FLAGS) $(INCLUDE) -c PostEnt2026/Examples/PythonModel.cpp
	$(CXX) -Wno-unused-variable -Wno-attributes -Wno-unused-parameter -DNO_THREADS $(FLAGS) $(INCLUDE) -c main.cpp
	$(CXX) $(LIBS) -o main main.o PythonModel.o -lpython3.12 -ldnest4 -pthread
	rm -f *.o
