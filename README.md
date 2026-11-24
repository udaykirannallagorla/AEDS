# Smart Marketplace Search and Real-Time Analytics System

A local Streamlit prototype based on the supplied KL University project description.

## Features
- Product catalogue with 1,000 seeded demonstration records
- Exact/substring, keyword and TF-IDF hybrid search
- Category, price-range and in-stock filters
- CSV catalogue import and stock/price updates with catalogue version increments
- Near-duplicate listing candidate detection
- Visitor-event dashboard with time-window filters
- Search algorithm benchmark and an idealized block/shard parallelism model
- Download search results as CSV

## Run on Windows
1. Install Python 3.10 or newer.
2. Open PowerShell in this project folder.
3. Create and activate a virtual environment:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If `py` is unavailable, use `python -m venv .venv` after installing Python and enabling it in PATH.
4. Install packages:
   ```powershell
   pip install -r requirements.txt
   ```
5. Start:
   ```powershell
   streamlit run app.py
   ```

## Data
- `data/products.csv`: seeded product catalogue
- `data/events.csv`: seeded visitor activity events

All demo data is synthetic. Importing a catalogue replaces the current CSV. Back up the data folder before importing.

## Notes / limitations
- TF-IDF is a lightweight local vector retrieval baseline, not a neural embedding model.
- Duplicate detection returns candidates; a human should review before merging.
- The block/shard calculation is an idealized teaching model, not a real distributed deployment.
- Search timings are machine-dependent. Run multiple trials and report median and spread for a reproducible evaluation.
- No live stock feed, real customers, payment processing, paid APIs or production deployment are included.
