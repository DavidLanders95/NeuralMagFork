convert-notebooks-to-scripts:
	jupyter nbconvert --to script --output-dir demos/generated docs/examples/*.ipynb --no-prompt
	isort demos/generated/*
	black demos/generated/*

.PHONY: convert-notebooks-to-scripts
