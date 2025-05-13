convert-notebooks-to-scripts:
	jupyter nbconvert --to script --output-dir demos docs/examples/*.ipynb --no-prompt
	isort demos/*
	black demos/*

.PHONY: convert-notebooks-to-scripts
