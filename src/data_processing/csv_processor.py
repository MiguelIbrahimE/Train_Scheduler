import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple, Union
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import requests
import json

class CSVProcessor:
    """Process and validate CSV files for railway route optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.geocoder = Nominatim(user_agent="railway_optimizer")
        
        # Standard column mappings
        self.column_mappings = {
            'city': ['city', 'name', 'city_name', 'place', 'location'],
            'lat': ['lat', 'latitude', 'y', 'lat_coord'],
            'lon': ['lon', 'lng', 'longitude', 'x', 'lon_coord'],
            'population': ['population', 'pop', 'inhabitants', 'people'],
            'country': ['country', 'nation', 'state'],
            'region': ['region', 'state', 'province', 'area'],
            'elevation': ['elevation', 'altitude', 'height'],
            'gdp': ['gdp', 'gdp_per_capita', 'income'],
            'area': ['area', 'area_km2', 'surface_area']
        }
    
    def detect_delimiter(self, file_path: str) -> str:
        """Detect CSV delimiter automatically"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            
            # Common delimiters
            delimiters = [',', ';', '\t', '|']
            delimiter_counts = {}
            
            for delimiter in delimiters:
                delimiter_counts[delimiter] = first_line.count(delimiter)
            
            # Return delimiter with highest count
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            
            if delimiter_counts[best_delimiter] == 0:
                return ','  # Default to comma
            
            return best_delimiter
            
        except Exception as e:
            self.logger.warning(f"Error detecting delimiter: {e}")
            return ','
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names based on common patterns"""
        
        df_copy = df.copy()
        
        # Convert all column names to lowercase
        df_copy.columns = df_copy.columns.str.lower()
        
        # Remove special characters and replace with underscores
        df_copy.columns = df_copy.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
        
        # Map columns to standard names
        column_mapping = {}
        
        for standard_name, variations in self.column_mappings.items():
            for col in df_copy.columns:
                for variation in variations:
                    if variation in col or col in variation:
                        column_mapping[col] = standard_name
                        break
                if col in column_mapping:
                    break
        
        # Apply mapping
        df_copy = df_copy.rename(columns=column_mapping)
        
        return df_copy
    
    def validate_city_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate city data and return cleaned DataFrame with error messages"""
        
        errors = []
        df_clean = df.copy()
        
        # Check required columns
        required_columns = ['city', 'lat', 'lon']
        missing_columns = [col for col in required_columns if col not in df_clean.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return pd.DataFrame(), errors
        
        # Validate and clean city names
        df_clean['city'] = df_clean['city'].astype(str).str.strip()
        df_clean = df_clean[df_clean['city'].notna() & (df_clean['city'] != '')]
        
        if df_clean.empty:
            errors.append("No valid city names found")
            return pd.DataFrame(), errors
        
        # Validate coordinates
        try:
            df_clean['lat'] = pd.to_numeric(df_clean['lat'], errors='coerce')
            df_clean['lon'] = pd.to_numeric(df_clean['lon'], errors='coerce')
        except Exception as e:
            errors.append(f"Error converting coordinates to numeric: {e}")
            return pd.DataFrame(), errors
        
        # Check coordinate ranges
        invalid_coords = (
            (df_clean['lat'] < -90) | (df_clean['lat'] > 90) |
            (df_clean['lon'] < -180) | (df_clean['lon'] > 180) |
            df_clean['lat'].isna() | df_clean['lon'].isna()
        )
        
        if invalid_coords.any():
            invalid_count = invalid_coords.sum()
            errors.append(f"Found {invalid_count} cities with invalid coordinates")
            df_clean = df_clean[~invalid_coords]
        
        # Validate population if present
        if 'population' in df_clean.columns:
            df_clean['population'] = pd.to_numeric(df_clean['population'], errors='coerce')
            df_clean['population'] = df_clean['population'].fillna(0).astype(int)
            
            # Check for unrealistic population values
            unrealistic_pop = (df_clean['population'] < 0) | (df_clean['population'] > 50000000)
            if unrealistic_pop.any():
                errors.append(f"Found {unrealistic_pop.sum()} cities with unrealistic population values")
                df_clean.loc[unrealistic_pop, 'population'] = 0
        
        # Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['city', 'lat', 'lon'])
        
        if len(df_clean) < initial_count:
            errors.append(f"Removed {initial_count - len(df_clean)} duplicate cities")
        
        return df_clean, errors
    
    def geocode_missing_coordinates(self, df: pd.DataFrame, country: str = None) -> pd.DataFrame:
        """Geocode cities with missing coordinates"""
        
        df_geocoded = df.copy()
        
        # Find cities with missing coordinates
        missing_coords = df_geocoded['lat'].isna() | df_geocoded['lon'].isna()
        
        if not missing_coords.any():
            return df_geocoded
        
        self.logger.info(f"Geocoding {missing_coords.sum()} cities with missing coordinates")
        
        for idx in df_geocoded[missing_coords].index:
            city_name = df_geocoded.loc[idx, 'city']
            
            # Build query
            query = city_name
            if country:
                query += f", {country}"
            
            try:
                location = self.geocoder.geocode(query, timeout=10)
                
                if location:
                    df_geocoded.loc[idx, 'lat'] = location.latitude
                    df_geocoded.loc[idx, 'lon'] = location.longitude
                    self.logger.info(f"Geocoded {city_name}: {location.latitude}, {location.longitude}")
                else:
                    self.logger.warning(f"Could not geocode {city_name}")
                
                # Rate limiting
                time.sleep(1)
                
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                self.logger.warning(f"Geocoding error for {city_name}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error geocoding {city_name}: {e}")
                continue
        
        return df_geocoded
    
    def enrich_city_data(self, df: pd.DataFrame, country: str = None) -> pd.DataFrame:
        """Enrich city data with additional information"""
        
        df_enriched = df.copy()
        
        # Add country if not present
        if 'country' not in df_enriched.columns and country:
            df_enriched['country'] = country
        
        # Estimate population if missing
        if 'population' not in df_enriched.columns:
            df_enriched['population'] = self._estimate_population(df_enriched)
        
        # Add regional information
        df_enriched = self._add_regional_info(df_enriched, country)
        
        # Calculate city importance score
        df_enriched['importance_score'] = self._calculate_importance_score(df_enriched)
        
        # Add economic indicators
        df_enriched = self._add_economic_indicators(df_enriched, country)
        
        return df_enriched
    
    def _estimate_population(self, df: pd.DataFrame) -> pd.Series:
        """Estimate population for cities without population data"""
        
        # Simple estimation based on city name patterns and country
        population_estimates = []
        
        for _, row in df.iterrows():
            city_name = row['city'].lower()
            
            # Basic estimation rules
            if any(keyword in city_name for keyword in ['new', 'north', 'south', 'east', 'west']):
                # Smaller cities
                pop = np.random.randint(50000, 200000)
            elif len(city_name) > 10:
                # Longer names often indicate smaller places
                pop = np.random.randint(20000, 100000)
            else:
                # Default medium size
                pop = np.random.randint(100000, 500000)
            
            population_estimates.append(pop)
        
        return pd.Series(population_estimates, index=df.index)
    
    def _add_regional_info(self, df: pd.DataFrame, country: str) -> pd.DataFrame:
        """Add regional information based on coordinates"""
        
        df_regional = df.copy()
        
        # Simple regional classification based on coordinates
        if country == 'germany':
            # Rough regional boundaries for Germany
            df_regional['region'] = df_regional.apply(
                lambda row: self._classify_german_region(row['lat'], row['lon']), axis=1
            )
        elif country == 'japan':
            # Japanese regions
            df_regional['region'] = df_regional.apply(
                lambda row: self._classify_japanese_region(row['lat'], row['lon']), axis=1
            )
        else:
            # Generic regional classification
            df_regional['region'] = 'unknown'
        
        return df_regional
    
    def _classify_german_region(self, lat: float, lon: float) -> str:
        """Classify German regions based on coordinates"""
        
        if lat > 53:
            return 'north'
        elif lat > 51:
            return 'central'
        elif lat > 49:
            return 'south_central'
        else:
            return 'south'
    
    def _classify_japanese_region(self, lat: float, lon: float) -> str:
        """Classify Japanese regions based on coordinates"""
        
        if lat > 43:
            return 'hokkaido'
        elif lat > 38:
            return 'tohoku'
        elif lat > 35:
            return 'kanto'
        elif lat > 33:
            return 'kansai'
        else:
            return 'kyushu'
    
    def _calculate_importance_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate importance score for cities"""
        
        scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Population score (normalized)
            if 'population' in row and pd.notna(row['population']):
                pop = row['population']
                if pop > 1000000:
                    score += 10
                elif pop > 500000:
                    score += 8
                elif pop > 200000:
                    score += 6
                elif pop > 100000:
                    score += 4
                elif pop > 50000:
                    score += 2
                else:
                    score += 1
            
            # Capital city bonus
            city_name = row['city'].lower()
            capitals = ['berlin', 'tokyo', 'paris', 'london', 'madrid', 'rome', 'vienna', 'bern', 'stockholm', 'copenhagen', 'amsterdam']
            if any(capital in city_name for capital in capitals):
                score += 5
            
            # Major city indicators
            major_indicators = ['central', 'main', 'principal', 'major']
            if any(indicator in city_name for indicator in major_indicators):
                score += 3
            
            scores.append(score)
        
        return pd.Series(scores, index=df.index)
    
    def _add_economic_indicators(self, df: pd.DataFrame, country: str) -> pd.DataFrame:
        """Add economic indicators to city data"""
        
        df_econ = df.copy()
        
        # GDP per capita estimates by country
        country_gdp = {
            'germany': 46259,
            'switzerland': 81867,
            'japan': 39285,
            'france': 38625,
            'netherlands': 52331,
            'austria': 45436,
            'sweden': 51925,
            'denmark': 60170
        }
        
        base_gdp = country_gdp.get(country, 35000)
        
        # Adjust GDP based on city size and importance
        df_econ['gdp_per_capita'] = df_econ.apply(
            lambda row: self._estimate_city_gdp(row, base_gdp), axis=1
        )
        
        # Tourism index (simplified)
        df_econ['tourism_index'] = df_econ.apply(
            lambda row: self._estimate_tourism_index(row), axis=1
        )
        
        # Economic activity index
        df_econ['economic_activity'] = df_econ.apply(
            lambda row: self._estimate_economic_activity(row), axis=1
        )
        
        return df_econ
    
    def _estimate_city_gdp(self, row: pd.Series, base_gdp: float) -> float:
        """Estimate GDP per capita for a city"""
        
        gdp = base_gdp
        
        # Adjust based on population (larger cities often have higher GDP)
        if 'population' in row and pd.notna(row['population']):
            pop = row['population']
            if pop > 1000000:
                gdp *= 1.3
            elif pop > 500000:
                gdp *= 1.2
            elif pop > 200000:
                gdp *= 1.1
            elif pop < 50000:
                gdp *= 0.9
        
        # Adjust based on importance score
        if 'importance_score' in row and pd.notna(row['importance_score']):
            score = row['importance_score']
            gdp *= (1 + score * 0.05)
        
        return gdp
    
    def _estimate_tourism_index(self, row: pd.Series) -> float:
        """Estimate tourism index for a city"""
        
        city_name = row['city'].lower()
        
        # Tourism hotspots
        tourist_cities = ['paris', 'london', 'tokyo', 'berlin', 'rome', 'barcelona', 'amsterdam', 'vienna', 'prague', 'zurich']
        
        if any(city in city_name for city in tourist_cities):
            return np.random.uniform(7, 10)
        
        # Medium tourism
        if 'population' in row and pd.notna(row['population']) and row['population'] > 200000:
            return np.random.uniform(4, 7)
        
        # Low tourism
        return np.random.uniform(1, 4)
    
    def _estimate_economic_activity(self, row: pd.Series) -> float:
        """Estimate economic activity index for a city"""
        
        activity = 5.0  # Base level
        
        # Adjust based on population
        if 'population' in row and pd.notna(row['population']):
            pop = row['population']
            if pop > 1000000:
                activity = 9.0
            elif pop > 500000:
                activity = 8.0
            elif pop > 200000:
                activity = 7.0
            elif pop > 100000:
                activity = 6.0
        
        # Add some randomness
        activity += np.random.uniform(-1, 1)
        
        return max(1.0, min(10.0, activity))
    
    def process_csv_file(self, file_path: str, country: str = None) -> Tuple[pd.DataFrame, List[str]]:
        """Process a single CSV file with full pipeline"""
        
        self.logger.info(f"Processing CSV file: {file_path}")
        
        try:
            # Detect delimiter
            delimiter = self.detect_delimiter(file_path)
            
            # Read CSV
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
            
            # Standardize columns
            df = self.standardize_columns(df)
            
            # Validate data
            df_validated, errors = self.validate_city_data(df)
            
            if df_validated.empty:
                return df_validated, errors
            
            # Geocode missing coordinates
            df_geocoded = self.geocode_missing_coordinates(df_validated, country)
            
            # Enrich data
            df_enriched = self.enrich_city_data(df_geocoded, country)
            
            self.logger.info(f"Successfully processed {len(df_enriched)} cities")
            
            return df_enriched, errors
            
        except Exception as e:
            error_msg = f"Error processing CSV file: {e}"
            self.logger.error(error_msg)
            return pd.DataFrame(), [error_msg]
    
    def process_multiple_csv_files(self, file_paths: List[str], countries: List[str] = None) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
        """Process multiple CSV files"""
        
        results = {}
        
        for i, file_path in enumerate(file_paths):
            country = countries[i] if countries and i < len(countries) else None
            
            df, errors = self.process_csv_file(file_path, country)
            
            # Use filename as key
            key = Path(file_path).stem
            results[key] = (df, errors)
        
        return results
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str, format: str = 'csv'):
        """Save processed data to file"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            df.to_csv(output_file, index=False)
        elif format.lower() == 'json':
            df.to_json(output_file, orient='records', indent=2)
        elif format.lower() == 'excel':
            df.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved processed data to {output_file}")
    
    def create_data_quality_report(self, df: pd.DataFrame, errors: List[str]) -> Dict:
        """Create a data quality report"""
        
        report = {
            'total_records': len(df),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'errors': errors,
            'statistics': {}
        }
        
        # Add statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report['statistics'][col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max(),
                'std': df[col].std()
            }
        
        # Add coordinate validation
        if 'lat' in df.columns and 'lon' in df.columns:
            report['coordinate_validation'] = {
                'valid_coordinates': ((df['lat'] >= -90) & (df['lat'] <= 90) & 
                                    (df['lon'] >= -180) & (df['lon'] <= 180)).sum(),
                'invalid_coordinates': ((df['lat'] < -90) | (df['lat'] > 90) | 
                                      (df['lon'] < -180) | (df['lon'] > 180)).sum()
            }
        
        return report
    
    def filter_by_bounding_box(self, df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
        """Filter cities by bounding box (south, west, north, east)"""
        
        if 'lat' not in df.columns or 'lon' not in df.columns:
            return df
        
        south, west, north, east = bbox
        
        filtered_df = df[
            (df['lat'] >= south) & (df['lat'] <= north) &
            (df['lon'] >= west) & (df['lon'] <= east)
        ]
        
        self.logger.info(f"Filtered from {len(df)} to {len(filtered_df)} cities within bounding box")
        
        return filtered_df
    
    def sort_cities_by_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort cities by importance score"""
        
        if 'importance_score' in df.columns:
            return df.sort_values('importance_score', ascending=False)
        elif 'population' in df.columns:
            return df.sort_values('population', ascending=False)
        else:
            return df.sort_values('city')

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create processor
    processor = CSVProcessor()
    
    # Test with sample data
    sample_data = {
        'city': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt'],
        'lat': [52.5200, 48.1351, 53.5511, 50.1109],
        'lon': [13.4050, 11.5820, 9.9937, 8.6821],
        'population': [3669491, 1471508, 1899160, 753056]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save sample data
    sample_file = "data/input/cities/germany/sample_cities.csv"
    Path(sample_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(sample_file, index=False)
    
    # Process the file
    processed_df, errors = processor.process_csv_file(sample_file, 'germany')
    
    if not processed_df.empty:
        print("\n=== Processed Cities Data ===")
        print(processed_df.to_string(index=False))
        
        # Create quality report
        report = processor.create_data_quality_report(processed_df, errors)
        print(f"\n=== Data Quality Report ===")
        print(f"Total records: {report['total_records']}")
        print(f"Columns: {report['columns']}")
        print(f"Errors: {report['errors']}")
        
        # Save processed data
        processor.save_processed_data(processed_df, "data/processed/cities/germany_processed.csv")
    
    else:
        print("No data processed")
        print(f"Errors: {errors}")