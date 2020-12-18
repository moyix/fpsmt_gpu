
all:
	@if [ ! -f theory.cu ]; then \
		./generate.sh sample_smt/const.smt2; \
	fi
	cmake -Bbuild -GNinja .
	cmake --build build

all-nodlto:
	cmake -Bbuild -GNinja -DUSEDLTO=NO .
	cmake --build build

clean:
	rm -f *.o SMTLIB/*.o smt
	rm -rf build cxx
