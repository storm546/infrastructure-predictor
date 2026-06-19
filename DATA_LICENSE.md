# Data License & Attribution

The procurement data used to build the dataset and train the model in this
repository comes from the **Open Contracting Data Registry — Bulgaria**, published
by **DIGIWHIST / opentender.eu** and mirrored by the Open Contracting Partnership:

- https://data.open-contracting.org/en/publication/44
- https://opentender.eu/bg

## License

The data is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International (CC BY-NC-SA 4.0)**.

https://creativecommons.org/licenses/by-nc-sa/4.0/

This means:

- **Attribution** — you must credit the source (done here and in the app footer).
- **NonCommercial** — you may not use the data for commercial purposes.
- **ShareAlike** — derivatives must be shared under the same license.

The raw OCDS files (`backend/data/raw/ocds/*.jsonl.gz`) are **not** committed to this
repository (see `.gitignore`); run `backend/fetch_data.py` to download them. The
derived `backend/data/processed/contracts_clean.csv` is a transformed subset shared
under the same CC BY-NC-SA 4.0 terms.
