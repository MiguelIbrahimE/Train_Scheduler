flowchart LR
    A[Raw data<br>(OSM, reports, DEM)] -->|ETL & cleaning| B(Feature store)
    B --> C[Graph construction<br>(stations = nodes, tracks = edges)]
    C --> D[Embed network<br>• node2vec<br>• GNN encoder]
    D --> E[Archetype clustering<br>(k-means / HDBSCAN)]
    D --> F[Supervised models<br>(e.g. XGBoost → predict<br>CAPEX , punctuality)]
    E & F --> G[Inference API:<br>"Given new corridor →<br>closest archetype + KPIs"]
