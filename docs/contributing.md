# Getting started with development

## Setup

*REL* is compatible with Python 3.7 or higher.

Clone the repository:

```console
git clone https://github.com/informagi/REL.git
```

Install using `virtualenv`:

```console
cd REL
python -m venv env
source env/bin/activate
python -m pip install -e .[develop]
```

Alternatively, install using Conda:

```console
cd REL
conda create -n REL python=3.8
conda activate REL
pip install -e .[develop]
```

## Running tests

REL uses [pytest](https://docs.pytest.org/) to run the tests. You can run the tests for yourself using:

```console
pytest
```

## Updating/Building the documentation

The documentation is written in [markdown](https://www.markdownguide.org/basic-syntax/), and uses [mkdocs](https://www.mkdocs.org/) to generate the pages.

To build the documentation for yourself:

```console
pip install -e .[docs]
mkdocs serve
```

You can find the documentation source in the [docs](https://github.com/informagi/REL/tree/main/docs) directory. 
If you are adding new pages, make sure to update the listing in the [`mkdocs.yml`](https://github.com/informagi/REL/blob/mkdocs/mkdocs.yml) under the `nav` entry.
