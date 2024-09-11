env:
	python3 -m venv .venv

install:
	pip install -r requirements.txt

startenv:
	source .venv/bin/activate

101:
	python3 langchain101/main.py
