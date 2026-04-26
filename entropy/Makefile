CXX = g++
FLAGS = -std=c++20 -O3 -march=native -Wall -Wextra -pedantic
INCLUDE = -I PostEnt2021 -I /usr/include/python3.12 -I ../fitting/pybind11/include

default:
	$(CXX) -Wno-unused-variable -Wno-attributes -Wno-unused-parameter $(FLAGS) -I . $(INCLUDE) -c PostEnt2021/Examples/PythonModel.cpp
	$(CXX) -Wno-unused-variable -Wno-attributes -Wno-unused-parameter $(FLAGS) -I . $(INCLUDE) -c main.cpp
	$(CXX) -o main main.o PythonModel.o -lpython3.12
	rm -f *.o
