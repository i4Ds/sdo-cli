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

notebook:
	( \
	  source bin/activate; \
 	  trap "deactivate; exit" INT; \
	  jupyter notebook; \
	)

visdom:
	( \
	  source bin/activate; \
	  trap "deactivate; exit" INT; \
	  python3 -m visdom.server -port 8080; \
	)