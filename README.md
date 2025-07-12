# Railway Network Planning AI System

An intelligent system that learns from efficient railway networks (like Switzerland, Belgium, Netherlands) and plans optimal railway networks for new regions using machine learning, terrain analysis, and cost optimization.

## 🚂 Features

### Core Capabilities
- **AI-Powered Planning**: Machine learning models trained on efficient railway networks
- **Terrain Analysis**: Intelligent routing considering mountains, valleys, water bodies
- **Cost Optimization**: Comprehensive cost modeling for construction and operations
- **Demand Modeling**: Multi-factor passenger demand prediction (workers, students, tourism)
- **Route Optimization**: Multiple route alternatives with pathfinding algorithms
- **Interactive Visualization**: Beautiful maps and charts showing planned networks

### Technical Highlights
- **Multi-Source Data Integration**: Population, elevation, tourism, economic data
- **Advanced Pathfinding**: NetworkX-based graph algorithms with terrain considerations
- **ML-Driven Efficiency**: Random Forest and Gradient Boosting models
- **Scalable Architecture**: Modular design supporting multiple countries/regions
- **Comprehensive Reporting**: Detailed analysis reports and visualizations

## 🏗️ Installation

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/YourUsername/Railway_AI_Planner.git
cd Railway_AI_Planner

# Run setup (installs dependencies and creates sample data)
python setup_and_run.py --setup

# Run example with sample data
python setup_and_run.py --example
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir input output data models

# Run the main system
python railway_planner.py --csv input/your_cities.csv --budget 1000000000
```

## 📊 Usage

### Basic Usage
```bash
# Plan network for your cities
python railway_planner.py --csv input/lebanon_cities.csv --budget 500000000

# Use the helper script
python setup_and_run.py --csv input/your_cities.csv --budget 1000000000
```

### Input Format
Create a CSV file with your cities:
```csv
city_id,city_name
LB-BEY,Beirut
LB-TRP,Tripoli
LB-SID,Sidon
```

### Advanced Configuration
Edit `config.py` to customize:
- Construction costs per kilometer
- Train specifications
- Terrain difficulty multipliers
- ML model parameters
- API keys for real data sources

## 🔬 How It Works

### 1. Data Collection
- **City Data**: Population, elevation, economic indicators
- **Terrain Analysis**: Elevation profiles, water crossings, urban areas
- **Demand Modeling**: Worker commuting, student travel, tourism flows

### 2. Machine Learning
- **Training Data**: Learns from efficient networks (Switzerland, Belgium, etc.)
- **Feature Engineering**: Distance, population, terrain, costs
- **Predictions**: Route efficiency, passenger satisfaction scores

### 3. Route Optimization
- **Graph Theory**: NetworkX-based pathfinding algorithms
- **Multi-Objective**: Balances cost, efficiency, and passenger demand
- **Alternatives**: Generates multiple route options for comparison

### 4. Cost Analysis
- **Construction**: Rail, electrification, stations, tunnels, bridges
- **Operations**: Train procurement, maintenance, energy
- **ROI Calculation**: Revenue vs. cost analysis

### 5. Visualization
- **Interactive Maps**: Folium-based maps with route details
- **Cost Charts**: Matplotlib visualizations of budget breakdown
- **Detailed Reports**: Comprehensive analysis documents

## 🗂️ Project Structure

```
Railway_AI_Planner/
├── railway_planner.py      # Main system implementation
├── setup_and_run.py       # Easy setup and execution script
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── input/                 # Input CSV files
│   ├── lebanon_cities.csv
│   └── europe_cities.csv
├── output/                # Generated reports and visualizations
├── data/                  # Cached data and models
└── models/                # Trained ML models
```

## 📈 Output Files

The system generates several output files:

### 1. Interactive Map (`railway_network_map.html`)
- **Visual Network**: See planned routes overlaid on real geography
- **Color Coding**: Different colors for underground, above-ground, on-ground sections
- **Interactive**: Click on routes and cities for detailed information
- **Legend**: Clear explanation of symbols and colors

### 2. Cost Analysis Chart (`cost_analysis.png`)
- **Route Costs**: Bar charts showing construction costs per route
- **Efficiency Metrics**: Cost per kilometer comparisons
- **Budget Breakdown**: Visual representation of budget allocation

### 3. Detailed Report (`network_analysis_report.txt`)
- **Executive Summary**: Key metrics and recommendations
- **Route Analysis**: Detailed breakdown of each planned route
- **Cost Breakdown**: Comprehensive cost analysis
- **ML Recommendations**: AI-suggested improvements

### 4. Network Data (`network_data.json`)
- **Raw Data**: Complete network data in structured format
- **API Integration**: Easy integration with other systems
- **Analysis Data**: All calculations and intermediate results

## 🧠 Machine Learning Components

### Learning from Efficient Networks
The system learns from successful railway networks:

- **Switzerland**: 99% electrification, 9.5/10 efficiency score
- **Belgium**: Dense network, 8.2/10 efficiency score  
- **Netherlands**: High frequency service, 8.8/10 efficiency score
- **Japan**: World-class punctuality, 9.8/10 efficiency score

### Model Types
- **Cost Predictor**: Random Forest for construction cost estimation
- **Demand Predictor**: Gradient Boosting for passenger demand
- **Efficiency Scorer**: Multi-factor efficiency evaluation

### Features Used
- Distance between cities
- Population served
- Terrain difficulty
- Number of stations
- Electrification ratio
- Average operating speed
- Service frequency
- Cost per kilometer

## 🌍 Real-World Applications

### Urban Planning
- **Smart Cities**: Integrate rail planning with urban development
- **Sustainability**: Reduce carbon emissions through efficient public transport
- **Economic Development**: Connect economic centers efficiently

### Infrastructure Investment
- **ROI Analysis**: Evaluate infrastructure investment opportunities
- **Risk Assessment**: Understand terrain and construction challenges
- **Budget Planning**: Optimize limited infrastructure budgets

### Policy Making
- **Transport Policy**: Evidence-based transport infrastructure decisions
- **Regional Development**: Connect underserved regions to economic centers
- **International Cooperation**: Cross-border rail network planning

## 🔧 Customization

### Adding New Data Sources
```python
# In railway_planner.py, modify DataCollector class
def _fetch_city_details(self, city_name, lat, lon):
    # Add your API calls here
    population = your_population_api(city_name)
    tourism = your_tourism_api(city_name)
    # ... 
```

### Custom Cost Models
```python
# In config.py, modify cost parameters
COSTS = {
    'rail_per_km': 7_000_000,  # Adjust based on your region
    'electrification_per_km': 1_500_000,
    # ...
}
```

### New Train Types
```python
# Add custom train specifications
TRAIN_TYPES['maglev'] = {
    'name': 'Magnetic Levitation',
    'capacity': 600,
    'max_speed': 500,
    'cost': 50_000_000,
    'high_speed': True
}
```

## 📚 Examples

### Lebanon Railway Network
```bash
python setup_and_run.py --csv input/lebanon_cities.csv --budget 500000000
```
Plans connections between Beirut, Tripoli, Sidon, Tyre, Zahle, and other Lebanese cities.

### European Network Extension
```bash
python railway_planner.py --csv input/europe_cities.csv --budget 2000000000
```
Plans high-speed connections between major European cities.

### Custom Region
1. Create your CSV with cities
2. Adjust budget based on region
3. Modify costs in `config.py` for local conditions
4. Run the planner

## 🚀 Future Enhancements

### Planned Features
- **Real-time Data**: Integration with live population/economic APIs
- **3D Visualization**: Three.js-based 3D network visualization
- **Mobile App**: Mobile interface for field planning
- **Multi-Modal**: Integration with bus, metro, and air transport
- **Climate Impact**: Carbon footprint analysis and optimization

### Research Areas
- **Deep Learning**: Neural networks for complex route optimization
- **Reinforcement Learning**: Dynamic scheduling and capacity planning
- **Computer Vision**: Satellite imagery analysis for terrain assessment
- **IoT Integration**: Real-time passenger flow optimization

## 🤝 Contributing

We welcome contributions! Areas where you can help:

- **Data Sources**: Add real API integrations
- **ML Models**: Improve prediction accuracy
- **Visualization**: Enhanced maps and charts
- **Performance**: Optimization for large networks
- **Documentation**: Tutorials and examples

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Swiss Federal Railways (SBB)**: Inspiration for efficient network design
- **OpenStreetMap**: Geographic data
- **NetworkX**: Graph analysis algorithms
- **Scikit-learn**: Machine learning framework
- **Folium**: Interactive mapping

## 📞 Support

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join GitHub Discussions for questions
- **Documentation**: Check the wiki for detailed guides
- **Email**: contact@railwayai.com (if you set up email)

---

*"Bringing cities back to people through intelligent railway network planning."*

🚂💡🌍 Happy Planning!