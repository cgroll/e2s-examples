.PHONY: run dry-run serve build-book

# Run the full pipeline (only rebuilds what's out of date)
run:
	uv run dvc repro

# Preview what would be executed without running anything
dry-run:
	uv run dvc repro --dry

# Serve the book locally with live-reload.
# NODE_OPTIONS forces IPv4-first DNS resolution: in this environment
# `localhost` resolves only to ::1, but myst's dev server self-fetches via
# 127.0.0.1, so without this it can bind fine but time out talking to itself
# (same fix VS Code's own remote server process uses).
serve:
	cd book && NODE_OPTIONS="--dns-result-order=ipv4first" uv run myst start

# Build a static HTML site into book/_build/html - no server needed to view
# it, download/scp that folder and open index.html directly. Same
# NODE_OPTIONS fix as `serve` (myst build also runs a self-fetch internally).
build-book:
	cd book && NODE_OPTIONS="--dns-result-order=ipv4first" uv run myst build --html
