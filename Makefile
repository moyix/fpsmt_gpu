
all:
	cmake -Bbuild -GNinja .
	cmake --build build

all-nodlto:
	cmake -Bbuild -GNinja -DUSEDLTO=NO .
	cmake --build build

clean:
	rm -f *.o SMTLIB/*.o smt
	rm -rf build cxx
