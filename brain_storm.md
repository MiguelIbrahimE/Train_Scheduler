1 · Big-picture goal

Build a reusable, data-driven reference model that captures the design DNA of six top-performing rail systems (Belgium, Netherlands, Switzerland, France, Germany, Japan).
When a new corridor alignment is supplied, the model will:

    Locate it in the learnt “rail design space” (which cluster / archetype it most resembles).

    Predict key KPIs – CAPEX per km, punctuality, train capacity, energy use, etc.

    Recommend design parameters inherited from the closest archetype (track class, signalling level, timetable philosophy).

2 · Key questions to answer

    What features actually matter?

        Geometry: curvature, gradient, tunnel %, track density.

        Operations: trains-per-track-km, punctuality (% < 5 min late), average headway.

        Capacity & demand: passenger-km/route-km, seat occupancy.

        Economics: € / track-km (surface, viaduct, tunnel), farebox recovery.

        Governance & planning: clock-face schedule (yes/no), infrastructure/ops split.

    At what resolution do we model?

        Country-level averages? -> too few samples.

        Line-level (e.g. each IC/HS corridor)?

        Segment-level (e.g. 10 km graph edges) – chosen so we have 50 000+ samples for ML.

    Which machine-learning recipe?

        Graph Neural Network encoder → dense embeddings (captures topology).

        Unsupervised clustering (HDBSCAN) → discover archetypes.

        Down-stream regressors (XGBoost) → KPI prediction.

    How to keep KPIs comparable across countries?

        Normalise punctuality to “+5 min arrival window”.

        Express costs in PPP-adjusted € 2025.

        Tag data confidence (annual report vs. scraped dashboard vs. estimated).

3 · Data shopping list
Theme	Candidate open sources	Quick notes
Track geometry, speed, electrification	OpenStreetMap (via osmnx), Eurostat GIS	Check tag completeness in JP & FR rural lines
DEM, terrain ruggedness	SRTM 30 m, ASTER GDEM	Topography → slope & tunnel cost penalties
Protected areas	WDPA, Natura 2000, UNESCO WHS	Used to flag high-risk alignment zones
Ops KPIs (punctuality, capacity)	Annual reports: NS, SBB, SNCF, JR Central, DB, Infrabel	Parse PDFs (Tabula / Camelot)
Economics	OECD rail infra price database, UIC stats	Convert to €/track-km, PPP 2025
Governance attributes	EU Fourth Railway Package docs, MLIT Japan	One-hot encode separation model, PSO coverage
4 · Prototype sprint plan (≈6 months)

    ETL & cleaning (4 wks) – pull geometry + KPI tables for the six countries; harmonise units, currencies, timestamps.

    Graph construction (2 wks) – stations as nodes, track segments as edges with features (speed, slope, curve).

    Feature engineering (2 wks) – derive density, tunnel %, trains/track-km, etc.

    Embedding & clustering (3 wks) – train GNN encoder, visualise clusters in UMAP 2-D.

    KPI regressors (2 wks) – XGBoost models for CAPEX, punctuality, capacity.

    Inference micro-service (3 wks) – FastAPI endpoint: upload GeoJSON → returns archetype label + predicted KPIs.

    Front-end visual (2 wks) – Leaflet/Deck.gl map + scatterplot so users see their corridor vs reference lines.

    Validation & demo (2 wks) – back-test on known upgrades (e.g. NL PHS programme, Swiss NRLA).

5 · Risks & mitigation

    Data gaps / inconsistent definitions → store raw values + metadata; expose filters in UI.

    Small N for ground-truth CAPEX → augment with historical projects, focus on cost ratios (tunnel vs. surface) not absolute numbers.

    Over-automation bias (e.g. too many tunnels) → enforce engineering heuristics (max 2 % grade, min R = 1800 m for 300 km/h).

    Stakeholder trust → publish methodology notebook, allow parameter tweaking, log provenance of each prediction.

6 · Stretch ideas

    Energy-use module – rough kWh/train-km estimate from speed profile + gradient.

    Freight overlay – tag segments suitable for 22.5 t axle-load and night-slot availability.

    Scenario generator – Monte Carlo tuning of cost & demand inputs to show uncertainty bands.

    Gamified UI – “drag corridor endpoints” and watch KPIs update live (use WebAssembly for speed).

Next action
Kick off ETL notebook — start with OpenStreetMap extracts for the Netherlands and Switzerland to prove feature pipeline.


