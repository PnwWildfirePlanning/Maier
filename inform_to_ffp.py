import urllib
import pandas as pd
import geopandas as gpd


inform_csv_dir = 'D:\\UPF_Inform_2020_2023.csv'
fdra_url ='https://services3.arcgis.com/T4QMspbfLg3qTGWY/ArcGIS/rest/services/PNW_FDRAs_public/FeatureServer/0/query?' + \
	'where=OBJECTID>-1&outFields=*&f=pgeojson&token='
out_csv = 'D:\\UPF_Inform_2020_2023_out.csv'
region_id_dict = {
	'Crest': 1,
	'Cottage Grove': 2,
	'North Umpqua': 3,
	'South Umpqua': 4}
fire_id_st_val = 1


csv_df = pd.read_csv(inform_csv_dir)
cols = ['IncidentName', 'FireDiscoveryDateTime', 'IncidentSize', 'Longitude', 'Latitude', 'Fire Cause General']
csv_df = csv_df [cols]
cause_dict = {
	'Natural': 1, 
	'Recreation and ceremony': 4,
	'Equipment and vehicle use': 2,
	'Debris and open burning': 5,
	'Missing data/not specified/undetermined': 9, 
	'Other causes': 9,
	'Arson/incendiarism': 7, 
	'Smoking': 3,
	'Power generation/transmission/distribution': 2,
	'Fireworks': 9,
	'Misuse of fire by a minor': 8, 
	'Railroad operations and maintenance': 6,
	'Firearms and explosives use': 9}
csv_df = csv_df.replace({'Fire Cause General': cause_dict})		

def geojson_rest(url):
	req = urllib.request.urlopen(url)
	gjson = req.read().decode('utf-8')
	gdf = gpd.read_file(gjson)
	return gdf 

r6fdra_gdf = geojson_rest(fdra_url)
proj_crs = r6fdra_gdf.crs
fires_gdf = gpd.GeoDataFrame(data=csv_df, geometry=gpd.points_from_xy(csv_df.Longitude, csv_df.Latitude)).set_crs(proj_crs)
fires_gdf = gpd.overlay(fires_gdf, r6fdra_gdf, how='intersection')
cols = ['IncidentName', 'FireDiscoveryDateTime', 'IncidentSize', 'Fire Cause General', 'FDRAName']
fires_df = fires_gdf[cols]
fires_df = fires_df.replace({'FDRAName': region_id_dict})
fires_df['FireId'] = range(fire_id_st_val, 1+len(fires_df))
cols = ['FDRAName', 'FireDiscoveryDateTime', 'FireId', 'IncidentSize', 'Fire Cause General', 'IncidentName']
cols_dict = {'FDRAName': 'RegionId', 'FireDiscoveryDateTime': 'FireDisc', 'IncidentSize': 'FireSize', 'Fire Cause General': 'FireCause', 'IncidentName': 'FireName'}
fires_df[cols].rename(cols_dict, axis=1).to_csv(out_csv)
