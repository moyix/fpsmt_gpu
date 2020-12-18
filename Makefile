
all:
	make --silent clean rng-CURAND
	make --silent clean rng-AES
	make --silent clean rng-CHAM
	make --silent clean

rng-%:
	cmake -Bbuild -GNinja -DRNG=$* .
	cmake --build build

all-nodlto:
	cmake -Bbuild -GNinja -DUSEDLTO=NO .
	cmake --build build

clean:
	rm -f *.o SMTLIB/*.o smt
	rm -rf build cxx
