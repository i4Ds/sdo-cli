setup: ## Sets up the dev environment
	./setup.sh

install:
	( \
	  source .venv/bin/activate; \
 	  cd src; \
	  pip install --editable .; \
	)

dev: 
	docker compose up 

dev-down:
	docker compose down --remove-orphans --volumes   

pip-freeze:
	pip freeze > requirements.txt

pip-list-outdated:
	pip list --outdated

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