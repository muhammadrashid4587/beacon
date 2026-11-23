.PHONY: test demo clean

test:
	python3 -m unittest discover -s tests -v

demo:
	python3 -m beacon.cli

live-demo:
	python3 -m beacon.tracking --demo

clean:
	rm -rf __pycache__ beacon/__pycache__ tests/__pycache__ *.db
