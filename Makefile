setup: ## Sets up the dev environment
	./setup.sh

install:
	( \
	  source .venv/bin/activate; \
	  pip install --editable .; \
	)

package:
	( \
	  source .venv/bin/activate; \
	  rm -rf ./dist; \
	  python setup.py sdist bdist_wheel; \
	)


test-publish:
	( \
	  source .venv/bin/activate; \
	  python -m twine upload --repository testpypi dist/*; \
	  python3 -m pip install --extra-index-url https://test.pypi.org/simple sdo-cli==0.0.3; \ 
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