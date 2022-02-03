TARGET = TenaciousTensor

# Source files
CPP_SOURCE =	./source/base.cpp \
		./source/operators.cpp \
		./source/convolution.cpp \
		./source/activ.cpp \

# Include files
INCLUDE = ./include \
              
OBJ = $(subst .cpp,.o,$(subst source,objects,$(CPP_SOURCE)))
CC = g++

# Compiler flags:
#   -g...: debugging
#   -Wall: enable all warning messages
#   -O2: speed!
CPP_FLAGS = -g -Wall -std=c++11 -O2

# Libraries
LIBRARIES = -lcblas

           
RM = rm -rf           

all : display $(TARGET)
	

display:
	@clear
	@mkdir -p objects
		
	@echo ".------------------------------------."
	@echo "| Compiling: TenaciousTensor ++      |#"
	@echo "| ---------                          |#"
	@echo "|                                    |#"
	@echo "| By: Prof. Carlo R. da Cunha, Ph.D. |#"
	@echo "|                                    |#"
	@echo "| Created: Jan/2022                  |#"
	@echo "| Revision: Fev/2022                 |#"
	@echo "'------------------------------------'#"
	@echo "  #####################################"
	@echo ""


$(TARGET): $(OBJ)
	

./objects/%.o: ./source/%.cpp
	@echo "# Building target: $<"
	$(CC) $< $(CPP_FLAGS) -I$(INCLUDE) -c -o $@ $(LIBRARIES)
	@echo ""
		
install: 
	@clear
	@echo "Creating static library..."
	@ar rcs lib$(TARGET).a objects/*.o
		
	@echo "# Copying headers ..."
	@cp include/tenten.hpp /usr/local/include/tenten
	@echo "# Copying binary..."
	@mv libTenaciousTensor.a /usr/local/lib
		
clean:
	@$(RM) ./objects/*.o $(TARGET) *~
	@rmdir objects
	@rm *.a
	
.PHONY: all clean	
	
