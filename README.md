# Suburb Properties Dashboard (Microburbs API)

A professional Streamlit dashboard that ingests Microburbs' suburb property listings API, cleans and normalizes the data, and presents an executive-friendly, interactive view with filters, KPIs, maps, detailed cards, and CSV export.

## Features
- Live data from Microburbs API (suburb parameterized)
- Robust normalization (dates, land/building size, numeric coercions)
- Executive KPIs (listings, median price, avg land size, etc.)
- Powerful filters (price, bedrooms, property type, listing date, search)
- Interactive map (pydeck) with tooltips
- Property cards with rich descriptions
- Download filtered results to CSV
- Caching and graceful error handling

## Quickstart

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Run the app

```bash
streamlit run app.py
```

3) Open the URL printed by Streamlit (typically http://localhost:8501)

## Configuration
- API base: `https://www.microburbs.com.au/report_generator/api/suburb/properties`
- Auth header: `Authorization: Bearer <token>`
  - Default token is `test` to match the sandbox example
  - You can override in the sidebar or set env var `MICROBURBS_TOKEN`

## Notes
- The API sometimes returns numeric fields as strings (e.g., `"605 m²"`, `"None"`, `"nan"`). The app normalizes these.
- `land_size` is parsed to square meters where units present. Missing/unknown values become NaN.
- `listing_date` is parsed to datetime.
- Coordinates are used for the map when available.

## Example endpoint

`https://www.microburbs.com.au/report_generator/api/suburb/properties?suburb=Belmont+North`

Use the sidebar to change suburb and refine filters.


