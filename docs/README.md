# AMT Documentation

This directory contains the documentation for the AMT project, built with [MkDocs](https://www.mkdocs.org/) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## Building the Documentation

To build the documentation, follow these steps:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the documentation:
   ```bash
   mkdocs build
   ```

3. Serve the documentation locally:
   ```bash
   mkdocs serve
   ```

4. Open your browser and navigate to [http://localhost:8000](http://localhost:8000)

## Documentation Structure

- `mkdocs.yml`: MkDocs configuration file
- `docs/`: Documentation source files
  - `index.md`: Home page
  - `overview/`: Overview of the project
  - `usage/`: Usage guides
  - `api/`: API reference
  - `development/`: Development guides

## Adding New Pages

To add a new page to the documentation:

1. Create a new Markdown file in the appropriate directory
2. Add the page to the navigation in `mkdocs.yml`

## Deploying the Documentation

The documentation is automatically deployed when changes are pushed to the main branch.

You can also manually deploy the documentation using [mike](https://github.com/jimporter/mike):

```bash
# Deploy the current version as "latest"
mike deploy --push --update-aliases latest

# Deploy a specific version
mike deploy --push v1.0.0
``` 