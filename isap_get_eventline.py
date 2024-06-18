#purpose: automate download of eventline proposed and planned features for isap analysis
#notes: requires rmat creds to get past nifc mfa
	#wfigs and nifs url good for 2024, will need to change for use subsequent years
#author: brian maier

import requests
from urllib import request
import urllib
import geopandas as gpd

#####-BEGIN CONFIG-#####
nifs_outfile_dir = 'D:' #where to save eventline shapefile
username = '' #rma un
password = '' #rma pwd
fires_select = [] #ex. 'PIONEER'
#####-END CONFIG-#####

def get_token_rma(username, password): #post method
	org_name = 'nifc'
	url = f'https://{org_name}.maps.arcgis.com/sharing/generateToken'
	username = 'RMA_Authoritative'
	password = 'RMA_1234!'
	payload = {'f': 'json',
		'username': username,
		'password': password,
		'referer': f'https://{org_name}.maps.arcgis.com/'}
	response = requests.request('POST', url, data=payload)
	return response.json()['token']
token_rma = get_token_rma(username, password)

def geojson_rest(url): #get method
	req = urllib.request.urlopen(url)
	gjson = req.read().decode('utf-8')
	gdf = gpd.read_file(gjson)
	return gdf

#get attributes from wfigs to filter eventline request, wfigs 2024 nwcc
def get_irwin():
	irwin_url = 'https://services3.arcgis.com/T4QMspbfLg3qTGWY/ArcGIS/rest/services/WFIGS_Incident_Locations_YearToDate/FeatureServer/0/query?' + \
	'where=GACC%3D%27NWCC%27+AND+IncidentTypeCategory%3D%27WF%27+AND+ContainmentDateTime+IS+NULL&fullText=&objectIds=&time=&geometry=&geometryType=esriGeometryEnvelope&inSR=&spatialRel=esriSpatialRelIntersects&resultType=none' + \
	'&distance=0.0&units=esriSRUnit_Meter&relationParam=&returnGeodetic=false&outFields=*&returnGeometry=true&featureEncoding=esriDefault' + \
	'&multipatchOption=xyFootprint&maxAllowableOffset=&geometryPrecision=&outSR=&defaultSR=&datumTransformation=&applyVCSProjection=false' + \
	'&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnExtentOnly=false&returnQueryGeometry=false&returnDistinctValues=false' + \
	'&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&returnZ=false&returnM=false' + \
	'&returnExceededLimitFeatures=true&quantizationParameters=&sqlFormat=none&f=pgeojson&token='
	df = geojson_rest(irwin_url)
	return df
irwin_df = get_irwin()

if len(fires_select) > 0:
	irwin_df = irwin_df[irwin_df.IncidentName.isin(fires_select)]
else:
	pass

irwin_filter = str(tuple(irwin_df.IrwinID.unique())).replace(' ', '')
if len(fires_select) == 1:
	irwin_filter = irwin_filter.replace(',', '')
else:
	pass

def get_eventline(token_rma):
	eventline_url = 'https://services3.arcgis.com/T4QMspbfLg3qTGWY/ArcGIS/rest/services/InternalView_NIFS_2024/FeatureServer/4/query?'
	data_dict = {'where': 'IRWINID IN ' + irwin_filter, 'f': 'pgeojson', 'outFields': '*', 'returnExceededLimitFeatures': 'true',
		'token': token_rma}
	eventline_data = urllib.parse.urlencode(data_dict).encode()
	req = request.Request(eventline_url, data=eventline_data)
	eventline_gdf = gpd.read_file(request.urlopen(req).read().decode('utf-8'))
	return eventline_gdf
eventline_gdf = get_eventline(token_rma)

#filter planned and proposed feature category types from nwcc eventline
eventline_pp = eventline_gdf.loc[(eventline_gdf.FeatureCategory.str.contains('Proposed')) | (eventline_gdf.FeatureCategory.str.contains('Planned'))]
#save as shapefile
eventline_pp.to_file(nifs_outfile_dir, driver='ESRI Shapefile')
