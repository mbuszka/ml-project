# Info
The report is in file `Report.pdf`.

This file is generated from `Report.ipynb`, which loads the results and generates plots.
It expects them to be in `results` directory.

To regenerate the `pdf` run:
- `jupyter nbconvert --no-input --execute --to pdf Report.ipynb`

# Data
To download data please use
- `bash get_data.sh small`
- `bash get_data.sh big`

# Experiments
To rerun experiments
- `python run_cf.py`
- `python run_svd.py`

Both scripts accept `--size` argument, the default is `small` and save the results into `csv` files in the working directory.