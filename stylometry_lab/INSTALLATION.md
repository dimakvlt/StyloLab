# StyloLab Installation Guide

## Quick Start (5 minutes)

### 1. Clone the Repository
```bash
git clone https://github.com/dimakvlt/StyloLab.git
cd StyloLab
```

### 2. Create Virtual Environment
```bash
# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Basic installation (for running the app):
pip install -r requirements.txt

# With development tools (for contributing/developing):
pip install -r requirements.txt -r requirements-dev.txt
```

### 4. Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 5. Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Detailed Setup

### Virtual Environment (Recommended)

A virtual environment isolates project dependencies from your system Python:

```bash
# Create
python -m venv venv

# Activate
source venv/bin/activate      # macOS/Linux
# or
venv\Scripts\activate         # Windows

# Deactivate (when done)
deactivate
```

### Installing Packages

```bash
# Install everything at once
pip install -r requirements.txt

# Install specific package
pip install streamlit==1.28.0

# Upgrade package
pip install --upgrade pandas

# Check installed packages
pip list

# See what's installed and where
pip show streamlit
```

### Troubleshooting

#### Issue: "No module named 'streamlit'"
**Solution:** Make sure virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

#### Issue: "NLTK punkt not found"
**Solution:** Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

#### Issue: "ImportError: No module named 'spacy'"
**Solution:** Install spacy and download model:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

#### Issue: "Permission denied" on macOS/Linux
**Solution:** Make scripts executable:
```bash
chmod +x venv/bin/activate
```

---

## Development Setup

For contributing or modifying code:

```bash
# Install everything including dev tools
pip install -r requirements.txt -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Check code formatting
black . --check

# Auto-format code
black .

# Run type checking
mypy analysis_engine.py

# Run linter
flake8 .
```

---

## Updating Dependencies

To update all packages to latest compatible versions:

```bash
pip install --upgrade -r requirements.txt
```

To freeze current versions (create a lock file):

```bash
pip freeze > requirements-lock.txt
```

---

## Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Install from requirements
pip install -r requirements.txt

# List installed packages
pip list

# Check for outdated packages
pip list --outdated

# Uninstall package
pip uninstall streamlit

# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv
```

---

## System Requirements

- **Python:** 3.9 or higher
- **OS:** macOS, Linux, or Windows
- **RAM:** 4GB minimum (8GB recommended for large documents)
- **Disk:** 500MB for dependencies + cache

---

## Verified Versions

This requirements.txt was tested with:
- Python 3.10.12
- macOS 14.x and Ubuntu 22.04 LTS
- Windows 11

---

## Need Help?

Check these resources:
1. **Streamlit Docs:** https://docs.streamlit.io
2. **NLTK Documentation:** https://www.nltk.org
3. **Scikit-learn Docs:** https://scikit-learn.org
4. **GitHub Issues:** https://github.com/dimakvlt/StyloLab/issues

---

**Last Updated:** June 2025
