.PHONY: build
build:
	@python setup.py sdist bdist_wheel

.PHONY: upload
upload: build
	@twine upload dist/*

.PHONY: gen
gen:
	@python -m lark.tools.standalone lang/specification.lark > graphlang/graph_parser.py

.PHONY: run
run:
	@python3 -m graphlang examples/choice.graph
