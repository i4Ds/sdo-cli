setup: ## Sets up the dev environment
	./setup.sh

install:
	( \
	  source .venv/bin/activate; \
 	  cd src; \
	  pip install --editable .; \
	)

pip-freeze:
	pip freeze > requirements.txt

notebook:
	( \
	  source .venv/bin/activate; \
 	  trap "deactivate; exit" INT; \
	  jupyter notebook; \
	)

visdom:
	( \
	  source .venv/bin/activate; \
	  trap "deactivate; exit" INT; \
	  python3 -m visdom.server -port 8080; \
	)