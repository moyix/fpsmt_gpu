
all: lab

lab: venv
	./venv/bin/jupyter lab --host 0.0.0.0

venv:
	virtualenv -p $(shell which python3) venv
	./venv/bin/pip3 install -r ./requirements.txt

clean:
	rm -rf venv
