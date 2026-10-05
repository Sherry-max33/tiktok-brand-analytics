setup:
	uv sync

crawl_users:
	python -m scripts.crawl_users

crawl_hashtags:
	python -m scripts.crawl_hashtags

build:
	python -m scripts.build_dataset

app_data:
	python scripts/export_app_data.py

test:
	pytest -q
