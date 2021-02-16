setup: ## Sets up the dev environment
	./setup.sh

install:
	( \
	  source bin/activate; \
 	  cd src; \
	  pip install --editable .; \
	)

pip-freeze:
	pip freeze > requirements.txt

# run: install
# 	../bin/data_loader args
