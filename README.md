# Bytebuddies-nhl-predictor

# Bytebuddies NHL Predictor - Kamk Student Project 2 Machine Learning.

This repository contains the data pipeline, exploratory notebooks, machine learning models, and project documentation for the Bytebuddies NHL Predictor project. 

The project uses a single monorepo structure managed with **uv** for Python dependencies, **Marimo** for reactive notebooks, and **Typst** (`kamk-thesis`) for project reporting.

## Project Structure

```text
Bytebuddies-nhl-predictor/
├── .venv/               # Virtual environment managed by uv (ignored by Git)
├── data/                # Local raw data and database files (ignored by Git)
├── logs/                # Application execution logs
├── notebooks/           # Marimo reactive notebooks (.py)
├── report/              # Typst report using the KAMK thesis template
├── .gitignore           # Git ignore rules
├── .python-version      # Pinned Python version for uv
├── main.py              # Main entry point script
├── pyproject.toml       # Project dependencies and configuration
├── README.md            # Project documentation
└── uv.lock              # Exact locked dependency versions
```

## Prerequisites

Before cloning and running the project, make sure you have the following installed on your machine:

### 1. Install `uv` (Python Package Manager)
We use `uv` to manage Python versions, virtual environments, and packages.
* **Linux / macOS:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Install the Carlito Font (Required for the Report)
The KAMK Typst template requires the **Carlito** font family to compile properly without font warnings.
* **Download directly (Windows / macOS / Linux):**
  1. Download the font family from [Google Fonts: Carlito](https://fonts.google.com/specimen/Carlito).
  2. Extract the `.zip` archive, select all `.ttf` files, and install them on your system (on Windows, right-click and choose **Install for all users**).
* **Ubuntu / Debian (via APT):**
  ```bash
  sudo apt update && sudo apt install fonts-crosextra-carlito
  ```
*(Note: Restart your code editor or terminal after installing the font so the compiler detects it.)*

### 3. Install Typst (For Compiling the Report)
You can use either a code editor extension or the standalone command-line compiler:
* **Option A (VS Code / Zed Extension):** Install the **Tinymist Typst** extension in your editor. It includes a built-in compiler and live PDF preview.
* **Option B (Typst CLI):**
  * **Linux (Snap):** `sudo snap install typst`
  * **Windows (Winget):** `winget install --id Typst.Typst`
  * **macOS (Homebrew):** `brew install typst`

---

## Getting Started (Initial Setup)

1. **Clone the repository:**
   ```bash
   git clone <REPOSITORY_URL>
   cd Bytebuddies-nhl-predictor
   ```

2. **Sync the Python environment:**
   Run the following command in the project root. It will automatically install the required Python version (defined in `.python-version`), create the `.venv` directory, and install all locked packages from `uv.lock`:
   ```bash
   uv sync
   ```

---

## Running the Tools

Because we use `uv`, you do not need to manually activate the virtual environment. Prefix commands with `uv run` from the project root to execute them inside the shared environment.

### 1. Marimo Notebooks (`notebooks/`)
Marimo notebooks are stored as pure Python `.py` files.
* **Open the Marimo dashboard to browse or edit notebooks:**
  ```bash
  uv run marimo edit
  ```
* **Create or edit a specific notebook:**
  ```bash
  uv run marimo edit notebooks/my_notebook.py
  ```
* **Run a notebook as an interactive web app (read-only view):**
  ```bash
  uv run marimo run notebooks/my_notebook.py
  ```

### 2. Main Entry Point (`main.py`)
To run the main execution script of the project using the shared virtual environment:
```bash
uv run main.py
```

### 3. Typst Report (`report/`)
The report is located in the `report/` directory and uses the `@preview/kamk-thesis:0.0.2` template.

* **Live compilation with CLI (watches for changes and opens the PDF automatically):**
  ```bash
  cd report
  typst watch thesis.typ --open
  ```
* **Single compilation to PDF:**
  ```bash
  typst compile report/thesis.typ report/thesis.pdf
  ```
* **Using VS Code / Zed:** Open `report/thesis.typ` and click **Preview** in the top-right corner (provided by the Tinymist extension).

---

## Adding New Python Packages

If you need to install a new Python library for the project, add it using `uv add` in the project root:
```bash
uv add <package-name>
```