# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a single Jupyter notebook data analysis project exploring whether U.S. retail sales (FRED RSXFS) can be a leading indicator for Walmart quarterly revenue. There are no backend services, databases, or frontend applications — just `analysis.ipynb` and CSV data in `data/`.

### Running the notebook

```bash
export PATH="$HOME/.local/bin:$PATH"
jupyter nbconvert --to notebook --execute analysis.ipynb --output /dev/null --ExecutePreprocessor.timeout=300
```

Or start JupyterLab for interactive use:

```bash
export PATH="$HOME/.local/bin:$PATH"
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

### Key gotchas

- **PATH**: `pip install --user` puts scripts in `~/.local/bin` which is not on PATH by default. Always prepend `export PATH="$HOME/.local/bin:$PATH"` or use full paths.
- **No `requirements.txt` in original repo**: Dependencies are only implicitly declared via imports. A `requirements.txt` was added for reproducibility.
- **CSV columns**: Both data CSVs use `date` and `value` as column names. The notebook renames them internally (e.g., `retail_sales`, `revenue_millions`, `quarter_end`).
- **Linting**: No linter configuration exists in the repo. Code lives entirely in `analysis.ipynb`.
- **Tests**: No automated test suite exists. Validation is done by executing the notebook end-to-end with `jupyter nbconvert --execute`.
