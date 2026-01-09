
#imports
print("Importing modules...")
import os
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import geopandas as gpd
import contextily as cx
import math
import numpy as np
import itertools
import requests
import json
import urllib
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from scipy.interpolate import interp1d
#from scipy.stats import percentileofscore
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.ensemble import IsolationForest
import pickle
from dataclasses import dataclass


project_crs = 4326
miles_to_meters = 1609.344


#change indices var names based on use_min_max
def make_nfdrs_names(use_min_max):
    d = {'ec': 'energy_release_component', 'bi': 'burning_index', 'sc': 'spread_component', 'ic': 'ignition_component', 'sfdi': 'severe_fire_danger_index'}
    if use_min_max == True:
        for key, value in d.items():
            d[key] += '_max'
    elif use_min_max == False:
        pass
    else:
        print("Fail...use_min_max must be True or False")
    return d

def user_vars_1(): 
    with open('user_vars_1.pkl', 'rb') as f:
        notebook_data_1 = pickle.load(f)
    user_input_1 = notebook_data_1['user_vars_1']
    user_input_1['nfdrs_names'] = make_nfdrs_names(user_input_1['use_min_max'])
    return user_input_1

def user_vars_2():   
    with open('user_vars_2.pkl', 'rb') as f:
        notebook_data_2 = pickle.load(f)
    user_input_2 = notebook_data_2['user_vars_2']
    return user_input_2

#function to return from fems api
#enter url in postman to fix/explore query as fems continues to change api
def returnGraphQl(query, variables):
    url = 'https://fems.fs2c.usda.gov/api/climatology/graphql'
    response = requests.post(url, json={'query': query, 'variables': variables})
    data = json.loads(response.text)
    return data

#datetime class for data retrieval in fems
#datetime will be current when Sig is init
class DatetimeNow(object):

    def __init__(self):
        self.now = datetime.now()#.astimezone(ZoneInfo(tz_local))
        self.next_week = self.now + timedelta(days=7)
        self.last_week = self.now + timedelta(days=-7)
        self.yesterday = self.now + timedelta(days=-1)
        self.tomorrow = self.now + timedelta(days=+1)
        self.today = self.now

#this bit is to create a percentile axis that aligns with values
#interpolation to capture all possible percentile values
#can also be used to translate back and forth from value to percentile
#need to edit, or account for when used outside of dual axis, to not allow return percentile greater than 100...
class PercentileConversion(object):
    
    def __init__(self, df_nfdrs_por, indice):
        self.df_nfdrs_por = df_nfdrs_por
        self.indice = indice
        self.percentile_dict = None

    def make_percentile_dict(self):
        s = self.df_nfdrs_por[self.indice].quantile(np.linspace(0.0, 1.0, 1001)) #maybe this should start at 0.1?
        df = pd.DataFrame(s.reset_index())
        df = df.rename({'index':'percentile', self.indice: 'value'}, axis=1)
        df['percentile'] = df['percentile'] * 100
        d = dict(zip(df['value'], df['percentile']))
        self.percentile_dict = d 
        
    def value_to_percent(self):
        x = list(self.percentile_dict.keys())
        y = list(self.percentile_dict.values())
        i = interp1d(x, y, fill_value='extrapolate')
        return i
    
    def percent_to_value(self):
        x = list(self.percentile_dict.keys())
        y = list(self.percentile_dict.values())
        i = interp1d(y, x, fill_value='extrapolate')
        return i            

#for making output dirs
def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

 
#station class to retrive data from fems
#use_min_max == True returns 'daily extremes' for nfdrs (not wx), else returns 1300 nfdrs
class Station(object):

    def __init__(self, stn_id, fuel_model):
        self.stn_id = stn_id
        self.fuel_model = fuel_model

    def get_fems_nfdrs(self, start_date, end_date, use_min_max):
        if use_min_max == False:
            query_nfdrs = """
            query NfdrsObs ($stationIds: String!, $fuelModels: String!, $startDateRange: Date!, $endDateRange: Date!) {
                nfdrsObs(
                    stationIds: $stationIds
                    fuelModels: $fuelModels
                    startDateRange: $startDateRange
                    endDateRange: $endDateRange
                    dateTimeFormat: LocalStationTime
                    startHour: 13
                    endHour: 13
                    hasHistoricData: ALL
                    ) {
                        data {
                            station_name
                            station_id
                            wrcc_id
                            latitude
                            longitude
                            elevation
                            observation_time
                            observation_time_lst
                            display_hour
                            display_hour_lst
                            nfdr_date
                            nfdr_time
                            nfdr_type
                            fuel_model
                            fuel_model_version
                            kbdi
                            one_hr_tl_fuel_moisture
                            ten_hr_tl_fuel_moisture
                            hun_hr_tl_fuel_moisture
                            thou_hr_tl_fuel_moisture
                            ignition_component
                            spread_component
                            energy_release_component
                            burning_index
                            herbaceous_lfi_fuel_moisture
                            woody_lfi_fuel_moisture
                            gsi
                            observation_type
                            quality_code
                    }
                }
            }
            """
            vars_nfdrs = {'stationIds': self.stn_id, 'fuelModels': self.fuel_model, 'startDateRange': start_date, 'endDateRange': end_date}
            json_nfdrs = returnGraphQl(query_nfdrs, vars_nfdrs)
            df_nfdrs = pd.DataFrame(json_nfdrs['data']['nfdrsObs']['data'])
        else:
            query_nfdrs = """
            query NfdrMinMax ($stationIds: String!, $fuelModels: String!, $startDate: Date!, $endDate: Date!) {
                nfdrMinMax(
                    stationIds: $stationIds
                    fuelModels: $fuelModels
                    startDate: $startDate
                    endDate: $endDate
                    hasHistoricData: ALL
                ) {
                    data {
                        station_name
                        station_id
                        wrcc_id
                        latitude
                        longitude
                        elevation
                        summary_date
                        nfdr_type
                        fuel_model
                        one_hr_tl_fuel_moisture_min
                        one_hr_tl_fuel_moisture_min_time
                        ten_hr_tl_fuel_moisture_min
                        ten_hr_tl_fuel_moisture_min_time
                        hun_hr_tl_fuel_moisture_min
                        hun_hr_tl_fuel_moisture_min_time
                        thou_hr_tl_fuel_moisture_min
                        thou_hr_tl_fuel_moisture_min_time
                        ignition_component_max
                        ignition_component_max_time
                        spread_component_max
                        spread_component_max_time
                        energy_release_component_max
                        energy_release_component_max_time
                        burning_index_max
                        burning_index_max_time
                        herbaceous_lfi_fuel_moisture
                        woody_lfi_fuel_moisture
                        gsi
                        kbdi
                        quality_code
                    }
                }
            }"""           
            vars_nfdrs = {'stationIds': self.stn_id, 'fuelModels': self.fuel_model, 'startDate': start_date, 'endDate': end_date}
            json_nfdrs = returnGraphQl(query_nfdrs, vars_nfdrs)
            df_nfdrs = pd.DataFrame(json_nfdrs['data']['nfdrMinMax']['data'])
        return df_nfdrs
        
#sig class for implementation of multiple stations using station class, should probably subclass station in sig...
#a single station can be used in sig class
class Sig(object):
    
    def __init__(self, stn_list, fuel_model, use_min_max):
        self.stn_list = stn_list
        self.fuel_model = fuel_model
        self.use_min_max = use_min_max 
        self.dt_now = DatetimeNow()
        self.user_vars = user_vars_1()
        self.por_st_dt  = str(self.user_vars['por_st_dt']) + '-01-01'
        self.por_end_dt = str(self.user_vars['por_end_dt']) + '-12-31'
        self.nfdrs_por = None
        self.interp_ec = None
        self.interp_bi = None

    #trim nfdrs_names por data to align between stations
    def align_dates(self, df):
        l_min = []
        l_max = []
        for stn_id in self.stn_list:
            dt_min = df.loc[df.station_id.astype('str') == stn_id]['datetime'].min()
            dt_max = df.loc[df.station_id.astype('str') == stn_id]['datetime'].max()
            l_min.append(dt_min)
            l_max.append(dt_max)
        sig_dt_min = max(l_min)
        sig_dt_max = min(l_max)
        res = df.loc[(df.datetime >= sig_dt_min) & (df.datetime <= sig_dt_max)]
        return res

    def calc_sfdi(self, df):
            df['ec_p'] = self.interp_ec.value_to_percent()(df[self.user_vars['nfdrs_names']['ec']])
            df['bi_p'] = self.interp_bi.value_to_percent()(df[self.user_vars['nfdrs_names']['bi']])
            if self.user_vars['use_min_max'] == False:
                df['severe_fire_danger_index'] = df.ec_p * df.bi_p
            else:
                df['severe_fire_danger_index_max'] = df.ec_p * df.bi_p
            return df
        
    def get_nfdrs(self):
        print("Getting Nfdrs...")
        start_date = self.por_st_dt
        end_date = self.por_end_dt
        l = []
        for stn_id in self.stn_list:
            df = Station(stn_id, self.fuel_model).get_fems_nfdrs(start_date, end_date, self.user_vars['use_min_max'])
            if self.user_vars['use_min_max'] == False:
                df['datetime'] = pd.to_datetime(df.display_hour_lst).dt.tz_localize(None)
                df['datetime'] = pd.to_datetime(df['datetime'].dt.date)
            else:
                df['datetime'] = pd.to_datetime(df.summary_date)      
            l.append(df)
        df = pd.concat(l, axis=0)
        df = self.align_dates(df)
        df['doy'] = df['datetime'].dt.dayofyear
        cols = df.select_dtypes('number').columns
        df_por = df.loc[(df.nfdr_type == 'O') & (df.datetime >= self.por_st_dt) & (df.datetime <= self.por_end_dt)] \
                .groupby('datetime')[cols].mean().reset_index()
        #use only por for calculation of percentiles
        ec_interp = PercentileConversion(df_por, self.user_vars['nfdrs_names']['ec'])
        ec_interp.make_percentile_dict()
        bi_interp = PercentileConversion(df_por, self.user_vars['nfdrs_names']['bi'])
        bi_interp.make_percentile_dict()
        self.interp_ec = ec_interp
        self.interp_bi = bi_interp
        self.calc_sfdi(df_por)
        if self.user_vars['use_min_max'] == False:
            sfdi_interp = PercentileConversion(df_por, 'severe_fire_danger_index')
            sfdi_interp.make_percentile_dict()
            df_por['severe_fire_danger_index'] = sfdi_interp.value_to_percent()(df_por.severe_fire_danger_index).round(1)
        else:
            sfdi_interp = PercentileConversion(df_por, 'severe_fire_danger_index_max')
            sfdi_interp.make_percentile_dict()
            df_por['severe_fire_danger_index_max'] = sfdi_interp.value_to_percent()(df_por.severe_fire_danger_index_max).round(1)            
        self.nfdrs_por = df_por
        

#analysis data
class AnalysisData(object):
    
    #get fire history and heat detection data
    def get_fod_fires(dir_fod_gdb, mask):
        print("Getting fires...")
        gdf = gpd.read_file(dir_fod_gdb, layer='Fires', mask=mask)
        gdf = gdf.to_crs(project_crs)
        gdf['datetime'] = gdf.DISCOVERY_DATE.str.split('T', expand=True)[0]
        gdf['datetime'] = pd.to_datetime(gdf['DISCOVERY_DATE'])
        return gdf
        
    def get_inform_fires(dir_inform_csv):
        df = pd.read_csv(dir_inform_csv, low_memory=False)
        df['datetime'] = pd.to_datetime(df['FireDiscoveryDateTime'], format='mixed').dt.date
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.rename({'IncidentSize': 'FIRE_SIZE'}, axis=1, inplace=True)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs=project_crs)
        return gdf
        
    def combine_fires(gdf_fod_fires, gdf_inform_fires):
        gdf_inform_fires['NWCG_GENERAL_CAUSE'] = gdf_inform_fires['Fire Cause General']
        cols = ['datetime', 'FIRE_SIZE', 'NWCG_GENERAL_CAUSE', 'geometry']
        gdf = pd.concat([gdf_fod_fires[cols], gdf_inform_fires[cols]], axis=0)
        return gdf
 
    def get_heat_detections(modis_or_viirs, tz_local):
        dt_now = DatetimeNow()
        
        def get_heat_year(year):
            def clean_heat(df):
                df['time'] = df.apply(lambda row: str(row.acq_time).zfill(4), axis=1)
                df['datetime'] = pd.to_datetime(df.acq_date + ' ' + df.time, utc=True)
                df['datetime'] = df.datetime.dt.tz_convert(tz_local).dt.tz_localize(None)
                df.drop(['acq_date', 'acq_time', 'time'], axis=1, inplace=True)
                return df 
            if modis_or_viirs == 'Viirs':
                df = pd.read_csv('https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/' + str(year) + '/viirs-snpp_' + str(year) + '_United_States.csv')
                df = clean_heat(df) 
            else:
                path = 'https://firms.modaps.eosdis.nasa.gov/data/country/modis/' + str(year) + '/modis_' + str(year) + '_United_States.csv'
                df = pd.read_csv('https://firms.modaps.eosdis.nasa.gov/data/country/modis/' + str(year) + '/modis_' + str(year) + '_United_States.csv')
                df = clean_heat(df)
            return df
            
        def get_heat():
            if modis_or_viirs == 'Modis':
                yr_st = 2005
            else:
                yr_st = 2012
            l = []
            for i in range(yr_st, dt_now.now.year + 1, 1):
                print("Getting heat detections " + str(i) + "...")
                try:
                    df = get_heat_year(i)
                    l.append(df)
                except Exception as e:
                    print(str(i) + " not available...")
            df = pd.concat(l)
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs=project_crs)
            return gdf
        gdf = get_heat()
        return gdf 
    
    #get weather stations (active, fire, perm) from WXx statbaseline api
    def get_raws():
        print("Getting raws...")
        url = 'https://weather.nifc.gov/ords/prd/wx/station/statbaseline/0'
        json_obj = json.loads(urllib.request.urlopen(url).read())['station_archive_baseline']
        df = pd.DataFrame(json_obj)
        df = df.loc[(df['Class'] == 'Permanent') & (df['Ownership Type'] == 'FIRE') & (df['Status'] == 'A')]
        df['Last Modified Date'] = pd.to_datetime(df['Last Modified Date (yyyymmdd hh24:mi:ss)'], errors='coerce')
        df.dropna(subset=['NWS ID'], inplace=True)
        df = df.sort_values(['Name', 'Last Modified Date']).drop_duplicates(subset=['NWS ID'], keep='last')
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude)).set_crs(project_crs)#.to_crs(project_crs)
        #remove leading zeros from station id, fems does not recognize (station id is an int in fems, not a str)
        gdf['NWS ID'] = gdf['NWS ID'].str.lstrip('0')    
        return gdf
    
    def __init__(self, gdf_fdra):
        self.fdra = gdf_fdra
        self.fires = None
        self.heat_detections = None
        self.raws = None
        self.user_vars = user_vars_1()
        
    def add_fires(self):
        gdf_fod_fires = AnalysisData.get_fod_fires(self.user_vars['dir_fod_gdb'], mask=self.fdra.dissolve())
        if self.user_vars['dir_inform_csv'] == None:
            res = gdf_fod_fires
        else:
            gdf_inform_fires = AnalysisData.get_inform_fires(self.user_vars['dir_inform_csv'])
            res = AnalysisData.combine_fires(gdf_fod_fires, gdf_inform_fires)
        self.fires = res

    def add_heat_detections(self):
        if self.user_vars['modis_or_viirs'] != None:
            res = AnalysisData.get_heat_detections(self.user_vars['modis_or_viirs'], self.user_vars['tz_local'])
            self.heat_detections = res
        else:
            self.heat_detections = None

    def add_raws(self):
        res = AnalysisData.get_raws()
        self.raws = res
     
                
class FdraAnalysisData(object):
    
    def group_heat_detections(heat_detections):
        gdf = heat_detections.copy()
        gdf['FIRE_SIZE'] = 1 #FIRE_SIZE to match fod fire data field, actually for sum of count
        gdf['datetime'] = pd.to_datetime(gdf.datetime.dt.date)
        res = gdf.groupby(gdf.datetime)[['FIRE_SIZE']].sum().reset_index()
        return res

    def __init__(self, gdf_fdra_select, analysis_data):
        self.analysis_data = analysis_data
        self.user_vars = user_vars_1()
        self.fdra = gdf_fdra_select
        self.fdop_name = self.fdra.ParentDocName.values[0]
        self.fdop_dir_name = ''.join(char for char in self.fdop_name if char.isalnum())
        self.fdra_name = self.fdra.FDRAName.values[0]
        self.fdra_dir_name = ''.join(char for char in self.fdra_name if char.isalnum())
        self.fires = gpd.clip(self.analysis_data.fires, self.fdra)
        if self.user_vars['modis_or_viirs'] != None:
            self.heat_detections = gpd.clip(self.analysis_data.heat_detections, self.fdra)
            self.heat_detections_grouped = FdraAnalysisData.group_heat_detections(self.heat_detections)
        else:
            self.heat_detections = None
            self.heat_detections_grouped = None
        self.raws = gpd.clip(self.analysis_data.raws.to_crs(32610), self.fdra.to_crs(32610).buffer(miles_to_meters*self.user_vars['fdra_buffer_raws'])).to_crs(project_crs)
        self.nfdrs_por = None
        self.raws_no_data = None
        self.output_dir = self.user_vars['cwd'] + self.fdop_dir_name + '\\' + self.fdra_dir_name
        make_dir(self.output_dir)


    def add_nfdrs_por(self):
        l = []
        l_no_fems_data = []
        for stn in self.raws['NWS ID'].unique():
            try:
                for fm in self.user_vars['fuel_models_interest']:
                    sig = Sig(stn_list=[stn], fuel_model=fm, use_min_max=self.user_vars['use_min_max'])
                    print(stn + " " + fm)
                    sig.get_nfdrs()
                    df_por = sig.nfdrs_por.copy()
                    df_por['fuel_model'] = fm
                    l.append(df_por)
            except Exception as e:
                print(stn + " failed...")
                l_no_fems_data.append(stn)
        res = pd.concat(l, axis=0)
        #change station_id back to string from float (result of using Sig class for indiv station)
        res['station_id'] = res.station_id.astype('int').astype('str')
        self.nfdrs_por = res
        self.raws_no_data = l_no_fems_data
        
    #need edit to show stations with no fems data red
    def make_station_map(self):
        print("Making station map...")
        plt.style.use('seaborn-v0_8-paper')
        fig, ax = plt.subplots()
        fdra = self.fdra.copy().to_crs(epsg=3857)
        raws = self.raws.copy().to_crs(epsg=3857)
        fdra.plot(ax=ax, color='none',edgecolor='k')
        raws.plot(ax=ax, marker='*', color='k')
        for x, y, label in zip(raws.geometry.x, raws.geometry.y, raws['NWS ID']):
            ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
        cx.add_basemap(ax, attribution_size=0, source=cx.providers.CartoDB.Positron)
        plt.xticks([])  # Remove x-axis ticks and labels
        plt.yticks([])
        plt.title(self.fdop_name + ' ' + self.fdra_name + '\nStations Used in ROC Analysis')
        plt.tight_layout()
        fig.savefig(self.output_dir + '\\RocStations_' + self.fdra_dir_name + '.jpg')

    def make_targets_map(self):
        print("Making targets map...")
        plt.style.use('seaborn-v0_8-paper')
        fig, ax = plt.subplots()
        fdra = self.fdra.copy().to_crs(epsg=3857)
        fires = self.fires.copy().to_crs(epsg=3857)
        fdra.plot(ax=ax, color='none', edgecolor='k')
        fires.plot(ax=ax, color='k', markersize=1)
        cx.add_basemap(ax, attribution_size=0, source=cx.providers.CartoDB.Positron)
        plt.xticks([])  # Remove x-axis ticks and labels
        plt.yticks([])
        plt.title(self.fdop_name + ' ' + self.fdra_name + '\nFires Used in ROC Analysis')
        plt.tight_layout()
        fig.savefig(self.output_dir + '\\RocTargets_Fires_' + self.fdra_dir_name + '.jpg')            
        if self.user_vars['modis_or_viirs'] != None:
            fig, ax = plt.subplots()
            heat_detections = self.heat_detections.copy().to_crs(epsg=3857)
            fdra.plot(ax=ax, color='none', edgecolor='k')
            heat_detections.plot(ax=ax, color='k', markersize=1)  
            cx.add_basemap(ax, attribution_size=0, source=cx.providers.CartoDB.Positron)
            plt.xticks([])  # Remove x-axis ticks and labels
            plt.yticks([])
            plt.title(self.fdop_name + ' ' + self.fdra_name + '\nHeat Used in ROC Analysis')
            plt.tight_layout()
            fig.savefig(self.output_dir + '\\RocTargets_Heat_' + self.fdra_dir_name + '.jpg')
        
        
class RocAnalysis(object):
    
    def filter_roc_results(results, min_class_spread):
        df = results.copy()
        df['opt_perc_diff'] = df['opt_perc_diff'].fillna(df['opt_perc'])
        def remove_negative_spread(df):
        	grp = df.groupby(by=['id'])['opt_perc_diff'].prod().reset_index()
        	neg = grp.loc[grp.opt_perc_diff <= 0][['id']]
        	df = df.assign(combined=df[['id']].agg(set, axis=1))
        	neg = neg.assign(combined=neg[['id']].agg(set, axis=1))
        	res = df[~df.combined.isin(neg['combined'])]
        	return res.drop(['combined'], axis=1)   
        df = remove_negative_spread(df)
        
        def remove_no_spread(df, min_class_spread):
            rmv = df.loc[df.level > 1]
            grp = rmv[['id', 'opt_perc_diff']].groupby(['id']).min()
            neg = grp.loc[grp.opt_perc_diff >= min_class_spread].reset_index()
            res = df.loc[df.id.isin(neg['id'])]        
            return res
        df = remove_no_spread(df, min_class_spread)
        return df

    def align_sig_dates(df):
        l_min = []
        l_max = []
        for stn_id in df.station_id.unique():
            dt_min = df.loc[df.station_id.astype('str') == stn_id]['datetime'].min()
            dt_max = df.loc[df.station_id.astype('str') == stn_id]['datetime'].max()
            l_min.append(dt_min)
            l_max.append(dt_max)
        sig_dt_min = max(l_min)
        sig_dt_max = min(l_max)
        #let the sig calculate mean for entire year even if one station in sig is down
        #perhaps the correct way to do this for analysis would be to deviate from wims/ff+
        #methods and remove all days from analysis where not all stations in sig reported...
        sig_dt_min = sig_dt_min.replace(month=1, day=1)
        sig_dt_max = sig_dt_max.replace(month=12, day=31)
        res = df.loc[(df.datetime >= sig_dt_min) & (df.datetime <= sig_dt_max)]
        return res
    
    def __init__(self, analysis_data):
        self.analysis_data = analysis_data
        self.user_vars_1 = user_vars_1()
        self.user_vars_2 = user_vars_2()
        self.results_temp = []
        self.results = []
        
        def update_user_vars():
            d1 = self.user_vars_1
            d2 = self.user_vars_2
            d1.update(d2)
            return d1
        self.user_vars = update_user_vars() 
        
        self.kfold_nsplits = self.user_vars['kfold_nsplits']
        
        #initial number of k folds, number of targets in the largest class rounded down to nearest 10
        def make_nsplits(target_percentiles):
            if self.user_vars['fires_or_heat'] == 'Fires':
                df = self.analysis_data.fires.copy()
            elif self.user_vars['fires_or_heat'] == 'Heat':
                df = self.analysis_data.heat_detections_grouped.copy()
            else:
                print("Options are Fire or Heat...")
                raise ValueError
            n_splits = len(df.loc[((df.FIRE_SIZE > round(np.percentile(df.FIRE_SIZE.dropna(), \
                target_percentiles[2]), 1)) & df.datetime.between(pd.Timestamp(str(self.user_vars['por_st_dt']) + '-01-01'), \
                pd.Timestamp(str(self.user_vars['por_end_dt']) + '-12-31')))].groupby('datetime')['FIRE_SIZE'].mean())
            print("There are " + str(n_splits) + " targets in the highest target class...") 
            if n_splits >= 10:
                n_splits = math.floor(n_splits / 10) * 10
            elif n_splits >= 5 and n_splits < 10:
                n_splits = 5
            else:
                n_splits = n_splits
            print("Using " + str(n_splits) + " folds for analysis...") 
            return n_splits   
            
        if self.user_vars['kfold_nsplits'] == 'auto':
            if self.user_vars['fires_or_heat'] == 'Fires':
                self.kfold_nsplits = make_nsplits(self.user_vars['fire_size_percentiles'])
            else:
                self.kfold_nsplits = make_nsplits(self.user_vars['heat_size_percentiles'])
                
        if self.user_vars['fires_or_heat'] == 'Fires':
            dir_roc_target = self.analysis_data.output_dir + '\\Roc_' + self.user_vars['fires_or_heat'] + '_' \
                + "".join(str(item) for item in self.user_vars['fire_size_percentiles'])
        else:
            dir_roc_target = self.analysis_data.output_dir + '\\Roc_' + self.user_vars['fires_or_heat'] + '_' \
                + "".join(str(item) for item in self.user_vars['heat_size_percentiles'])
        make_dir(dir_roc_target)
        
        if self.user_vars['contamination'] == None:
            contamination_label = '_0'
        elif self.user_vars['contamination'] == 'auto':
            contamination_label = '_auto'
        else:
            contamination_label = '_' + str(int(self.user_vars['contamination']*10)) 

        self.output_dir_roc = dir_roc_target + '\\Contamination' + contamination_label + '\\Folds_' + str(self.kfold_nsplits) + '\\'
        make_dir(self.output_dir_roc)    

        if self.user_vars['fires_or_heat'] == 'Fires':
            #target list, 5 classes, need to change below to accomodate more/less, should change to without manual edits
            self.f_acres = [round(np.percentile(self.analysis_data.fires.FIRE_SIZE.dropna(), self.user_vars['fire_size_percentiles'][0]), 1), \
                             round(np.percentile(self.analysis_data.fires.FIRE_SIZE.dropna(), self.user_vars['fire_size_percentiles'][1]), 1), \
                             round(np.percentile(self.analysis_data.fires.FIRE_SIZE.dropna(), self.user_vars['fire_size_percentiles'][2]), 1)]
            self.f_acres.append(self.f_acres[-1])
            self.f_percentiles = self.user_vars['fire_size_percentiles']
        elif self.user_vars['fires_or_heat'] == 'Heat':
            self.f_acres = [round(np.percentile(self.analysis_data.heat_detections_grouped.FIRE_SIZE, self.user_vars['heat_size_percentiles'][0]), 1), \
                             round(np.percentile(self.analysis_data.heat_detections_grouped.FIRE_SIZE, self.user_vars['heat_size_percentiles'][1]), 1), \
                             round(np.percentile(self.analysis_data.heat_detections_grouped.FIRE_SIZE, self.user_vars['heat_size_percentiles'][2]), 1)]
            self.f_acres.append(self.f_acres[-1])
            self.f_percentiles = self.user_vars['heat_size_percentiles']
        else:
            print("Options are Fires or Heat...")
            raise ValueError
            
        #make sig data (nfdrs por) combinations for all stations combinations
        def make_sig_combinations(nfdrs_por):
            
            #make all possible station combinations from stations in fdra
            def make_stn_combinations(stn_list): #list of stations
                all_combinations = []
                for r in range(len(stn_list) + 1):
                    combinations = itertools.combinations(stn_list, r)
                    combinations_list = list(combinations)
                    all_combinations += combinations_list
                #sigs of 1-3 stations only
                l = [x for x in all_combinations if (len(x) <= 3) and (len(x) > 0)]
                return l
            
            stn_combinations = make_stn_combinations(nfdrs_por.station_id.unique())
            l = []
            count = 1
            for sig in stn_combinations:
                sig_nfdrs = nfdrs_por[nfdrs_por.station_id.isin(list(sig))]
                for fm in self.user_vars['fuel_models_interest']:
                    sig_nfdrs_fm = sig_nfdrs.loc[sig_nfdrs.fuel_model == fm]
                    if sig_nfdrs_fm.empty == False:
                        sig_nfdrs_fm = RocAnalysis.align_sig_dates(sig_nfdrs_fm)
                        cols = sig_nfdrs_fm.select_dtypes('number').columns
                        res = sig_nfdrs_fm.groupby('datetime')[cols].mean().reset_index()
                        res['fuel_model'] = fm
                        res['sig_id'] = count
                        res['sig_stns'] = ''
                        for i in res.index:
                            res.at[i, 'sig_stns'] = sig
                        l.append(res)
                count += 1
            return l
            
        self.sig_combinations = make_sig_combinations(self.analysis_data.nfdrs_por)

    def roc_calc_kfold(self, indice):
        
        indice = self.user_vars['nfdrs_names'][indice]
        
        #prep analysis data for roc 
        def prep_roc_data(sig_combination):

            def prep_data_st_end(df_wx, df_fires):
                min_dates = []
                max_dates = []
                min_dates.append(df_wx.datetime.min())
                max_dates.append(df_wx.datetime.max())
                min_dates.append(df_fires.datetime.min())
                max_dates.append(df_fires.datetime.max())
                dates = (max(min_dates), min(max_dates))
                dates = (dates[0].replace(month=1, day=1), dates[1].replace(month=12, day=31))
                return dates

            def prep_data_align_dates(df, data_dates):
                return df.loc[(df.datetime >= data_dates[0]) & (df.datetime <= data_dates[1])]

            df_wx = sig_combination.copy()
            if self.user_vars['fires_or_heat'] == 'Fires':
                df_fires = self.analysis_data.fires.copy()
            else:
                df_fires = self.analysis_data.heat_detections_grouped.copy()
            data_dates = prep_data_st_end(df_wx, df_fires)
            df_wx = prep_data_align_dates(df_wx, data_dates)
            df_fires = prep_data_align_dates(df_fires, data_dates)
            
            df = pd.merge(df_fires, df_wx, how='right', left_on=['datetime'], right_on=['datetime'])
            df.loc[df['FIRE_SIZE'] > self.f_acres[2], 'y'] = 4
            df.loc[df['FIRE_SIZE'] <= self.f_acres[2], 'y'] = 3
            df.loc[df['FIRE_SIZE'] <= self.f_acres[1], 'y'] = 2
            df.loc[df['FIRE_SIZE'] <= self.f_acres[0], 'y'] = 1
            df.loc[df['y'].isna(), 'y'] = 0     
            df['y'] = df.y.astype(int)
            df = df[~df[indice].isna()]
            res = df.sort_values(['datetime', 'y'], ascending=True).drop_duplicates(['datetime'], keep='last')
            return res
      
        cmap = ['#d7191c','#fdae61','#ffffbf','#2c7bb6'][::-1]
        n_splits = self.kfold_nsplits
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        count = 1
        for sig_combination in self.sig_combinations:
            
            indice_interp = PercentileConversion(sig_combination, indice)
            indice_interp.make_percentile_dict()
            
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), edgecolor=None, linewidth=1)
                plt.style.use('seaborn-v0_8-paper')
                print(' '.join(sig_combination.sig_stns.values[0]) + ' ' + sig_combination.fuel_model.values[0])
                df_target = prep_roc_data(sig_combination)
                sig_fpr = []
                sig_tpr = []
                sig_opt = []
                sig_auc = []
                sig_pr = []
                
                X = df_target[indice] #feature or dependent variable
                X = X.values.reshape(-1, 1) #reshape only if single indice
                y = df_target['y'].values #target variable classes=[0,1,2,3,4]
                y_n = np.bincount(y)
                mean_fpr = np.linspace(0, 1, 100)
                
                #remove outliers,testing, https://scikit-learn.org/stable/modules/outlier_detection.html
                def remove_outliers_isoforest(X, y):
                    l = []
                    df = pd.DataFrame({'X': [i for sublist in X for i in sublist], 'y': y})
                    ##print(df)
                    for i in [1,2,3,4]:
                        df_i = df.loc[df.y == i]
                        iso_forest = IsolationForest(contamination=self.user_vars['contamination'], random_state=42)
                        outlier_mask = iso_forest.fit_predict(df_i.X.values.reshape(-1, 1)) != -1
                        l.append(df_i[outlier_mask])
                    l.append(df.loc[df.y == 0])
                    return(pd.concat(l))

                if self.user_vars['contamination']: 
                    df_clean = remove_outliers_isoforest(X, y)
                    X = df_clean.X 
                    X = X.values.reshape(-1, 1) 
                    y = df_clean.y.values 
                    y_n = np.bincount(y)
                
                #cross fold validation alternative to test train split
                for i, (train, test) in enumerate(cv.split(X, y)):	
                    fold_fpr = dict()
                    fold_tpr = dict()
                    fold_roc_auc = dict()
                    fold_x_opt = dict()
                    fold_y_opt = dict()
                    fold_opt_v = dict()
                    ##print("Fold: " + str(i))
                    #scale data for learning to 0 mean and unit variance
                    scaler = StandardScaler().fit(X[train])
                    #create and train test model
                    model = SGDClassifier(loss='log_loss', shuffle=False, class_weight=None)
                    model.fit(scaler.transform(X[train]), y[train])
                    y_pred = model.predict(scaler.transform(X[test]))
                    probs = model.predict_proba(scaler.transform(X[test]))
                    for i, clr in zip(range(1,5,1), cmap): #0 is no fires
                        idx_i = np.where(y[test]==i)
                        idx_null = np.where(y[test]==0)
                        idx = np.concatenate([idx_i, idx_null], axis=1)
                        y_true = np.take(y[test], idx)
                        y_true[np.where(y_true == i)] = 1 #convert from class to binary
                        y_scores = np.take(probs[:, 1], idx) #keep positive scores only
                        X_test = np.take(X[test], idx)
                        fold_fpr[i], fold_tpr[i], thresholds = roc_curve(y_true[0], y_scores[0])
                        fold_roc_auc[i] = roc_auc_score(y_true[0], y_scores[0])
                        t_opt = thresholds[np.argmax(fold_tpr[i] - fold_fpr[i])] #optimal value
                        fold_x_opt[i] = fold_fpr[i][np.argmax(fold_tpr[i] - fold_fpr[i])]
                        fold_y_opt[i] = fold_tpr[i][np.argmax(fold_tpr[i] - fold_fpr[i])]
                        fold_opt_v[i] = X_test[0].ravel()[y_scores[0] == t_opt][0]
                        ##print(fold_opt_v[i]) 
                        ax1.plot(fold_fpr[i], fold_tpr[i], lw=0.5, alpha=0.5, color=clr)
                        ax1.plot(np.linspace(0,1,num=100), np.linspace(0,1,num=100),
                            color='0.5', lw=0.5, linestyle='dashed')
                        ax1.set_xlim([0.0, 1.0])
                        ax1.set_ylim([0.0, 1.05])
                        ax1.set_xlabel("False Positive Rate")
                        ax1.set_ylabel("True Positive Rate")
                        ax1.set_aspect('equal')	
                        ax1.scatter(fold_x_opt[i], fold_y_opt[i], marker='*', color=clr, alpha=0.5)					
                        ax1.set_title("ROC Stratified " + str(n_splits) + "-fold Cross Validation")
                    sig_fpr.append(fold_fpr)
                    sig_tpr.append(fold_tpr)
                    sig_opt.append(fold_opt_v)
                    sig_auc.append(fold_roc_auc)
                ##print("Adding fold rates")
                fprs1 = [i[1] for i in sig_fpr]
                fprs2 = [i[2] for i in sig_fpr]
                fprs3 = [i[3] for i in sig_fpr]
                fprs4 = [i[4] for i in sig_fpr]
                tprs1 = [i[1] for i in sig_tpr]
                tprs2 = [i[2] for i in sig_tpr]
                tprs3 = [i[3] for i in sig_tpr]
                tprs4 = [i[4] for i in sig_tpr]
                opt1 = np.mean([i[1] for i in sig_opt])
                opt2 = np.mean([i[2] for i in sig_opt])
                opt3 = np.mean([i[3] for i in sig_opt])
                opt4 = np.mean([i[4] for i in sig_opt])
                aucs1 = np.mean([i[1] for i in sig_auc])
                aucs2 = np.mean([i[2] for i in sig_auc])
                aucs3 = np.mean([i[3] for i in sig_auc])
                aucs4 = np.mean([i[4] for i in sig_auc])
                X_ = list(itertools.chain(*X))
                #changed all percentileofscore to use class conversion for continuity
                #s4 = percentileofscore(X_, opt4) - percentileofscore(X_, opt3)
                s1 = indice_interp.value_to_percent()(opt1).item() - 0
                s2 = indice_interp.value_to_percent()(opt2).item() - indice_interp.value_to_percent()(opt1).item()
                s3 = indice_interp.value_to_percent()(opt3).item() - indice_interp.value_to_percent()(opt2).item()
                s4 = indice_interp.value_to_percent()(opt4).item() - indice_interp.value_to_percent()(opt3).item()
                interp_tprs1 = []
                interp_tprs2 = []
                interp_tprs3 = []
                interp_tprs4 = []
                for i in range(n_splits):
                    interp_tprs1.append(np.interp(mean_fpr, fprs1[i], tprs1[i]))
                    interp_tprs2.append(np.interp(mean_fpr, fprs2[i], tprs2[i]))
                    interp_tprs3.append(np.interp(mean_fpr, fprs3[i], tprs3[i]))
                    interp_tprs4.append(np.interp(mean_fpr, fprs4[i], tprs4[i]))
                interp_tprs1 = np.mean(interp_tprs1, axis=0)
                interp_tprs2 = np.mean(interp_tprs2, axis=0)
                interp_tprs3 = np.mean(interp_tprs3, axis=0)
                interp_tprs4 = np.mean(interp_tprs4, axis=0)
                mean_tprs = [interp_tprs1, interp_tprs2, interp_tprs3, interp_tprs4]
                roc_opt_list = [opt1, opt2, opt3, opt4]
                spaces = [s1, s2, s3, s4]
                roc_auc_list = [aucs1, aucs2, aucs3, aucs4]
                f_label = ['<=' + str(self.f_acres[0]), '>' + str(self.f_acres[0]) + ' <=' + str(self.f_acres[1]), \
                    '>' + str(self.f_acres[1]) + ' <=' + str(self.f_acres[2]), '>' + str(self.f_acres[2])]
                if indice != self.user_vars['nfdrs_names']['sfdi']: #already a percentile of a percentile
                    percentiles = [indice_interp.value_to_percent()(opt1).item(), indice_interp.value_to_percent()(opt2).item(), \
                        indice_interp.value_to_percent()(opt3).item(), indice_interp.value_to_percent()(opt4).item()]
                else:
                    percentiles = [opt1, opt2, opt3, opt4]
                for i, opt, auc, acre, clr, perc in zip(range(4), roc_opt_list, roc_auc_list, f_label, cmap, percentiles):
                    ax1.plot(mean_fpr, mean_tprs[i], color='0.0', lw=1)
                    x_opt = mean_fpr[np.argmax(mean_tprs[i] - mean_fpr)]
                    y_opt = mean_tprs[i][np.argmax(mean_tprs[i] - mean_fpr)]
                    n = y_n[i + 1]
                    ##print(opt, perc)
                    ax1.scatter(x_opt, y_opt, marker='*', s=120, color=clr, edgecolor='0.0', linewidth=1.0,
                        label= "Target: " + str(acre) + "\nOptimum Value: {:.1f}".format(np.mean(opt)) + ' (' + str(round(perc, 1)) + '%)' + \
                        "\nMean AUC: {:.3f}".format(np.mean(auc)) + "\nn=" + str(n))
                    ax1.legend(loc="lower right")
                res = pd.DataFrame({'target': self.f_acres, 'opt_value': roc_opt_list, 'auc': roc_auc_list})
                res['fdra'] = self.analysis_data.fdra_name
                res['stations'] = ' '.join(df_target.sig_stns.values[0])
                res['fuel_model'] = df_target.fuel_model.values[0]
                res['indice'] = indice
                res['opt_perc'] = res['opt_value'].apply(lambda x: round(indice_interp.value_to_percent()(x).item(), 1))#round(percentileofscore(df_target[indice], x), 1))
                res['opt_perc_diff'] = res['opt_perc'].diff()
                res['id'] = count
                res['level'] = [1, 2, 3, 4]
                self.results_temp.append(res)
                try:
                    cmap2 = ['#d7191c','#fdae61','#ffffbf','#abd9e9','#2c7bb6'][::-1]
                    roc_opt_list.insert(4, df_target[indice].max() + 1)
                    roc_opt_list.insert(0, -1)
                    df_target['Level'] = pd.cut(df_target[indice], bins=roc_opt_list, labels=[1,2,3,4,5])
                    df_target['Count'] = 1
                    df_target['Month'] = df_target.datetime.dt.month
                    crosstab = pd.crosstab(index=df_target.Month, columns=df_target.Level, values=df_target.Count, aggfunc='sum', normalize=0)                
                    crosstab.plot.bar(stacked=True, ax=ax2, width=1, color=cmap2, edgecolor='k', alpha=0.8)
                    #ax2.xticks(rotation=0)
                    ax2.set_title('Percent of Days by Month & Level')
                    ax2.set_ylabel('Percent')
                    sns.histplot(data=df_target.loc[df_target.y > 0], ax=ax3, stat='count', multiple='stack', x='Level', hue='y', legend=True, palette=cmap, binrange=(1,5))
                    cmap3 = ['#d7191c','#fdae61'][::-1]
                    sns.histplot(data=df_target.loc[df_target.y > 2], ax=ax4, stat='count', multiple='stack', x='Level', hue='y', legend=True, palette=cmap3, binrange=(1,5))
                    ax3.set_title('Count of ' + self.user_vars['fires_or_heat'] + ' by Target Class (1-4) and Level')
                    ax4.set_title('Count of ' + self.user_vars['fires_or_heat'] + ' by Target Class (3 & 4) and Level')
                    ax3.legend(title='Target Class', labels=f_label[::-1])
                    ax4.legend(title='Target Class', labels=f_label[::-1])
                except ValueError:
                    pass
                fig.suptitle(self.analysis_data.fdop_name + ' ' + self.analysis_data.fdra_name + '\nIndice: ' + indice + ' & Fuel Model: ' + df_target.fuel_model.values[0] \
                        + ' & Contamination: ' + str(self.user_vars['contamination']) + '\nStations: ' + ' '.join(df_target.sig_stns.values[0]) \
                        + ' & Analysis Years: ' + str(df_target.datetime.dt.year.min()) + '-' + str(df_target.datetime.dt.year.max()) \
                        + '\nTarget: ' + self.user_vars['fires_or_heat'] +  ' & Target Percentiles: ' + ','.join([str(i) for i in self.f_percentiles]))
                fig.tight_layout()
                plt.savefig(self.output_dir_roc + str(count) + '_' + indice + '.jpg')
                count += 1
                plt.close('all')
            except Exception as e:
                print(e)
                count+=1
                pass
                
        try:
            results_cols = ['fdra', 'id', 'stations', 'fuel_model', 'indice', 'level', 'target', 'auc', 'opt_value', \
                                'opt_perc', 'opt_perc_diff']       
            results_indice =  pd.concat(self.results_temp, axis=0)[results_cols]
            results_indice.to_csv(self.output_dir_roc + 'RocResults_' + indice + '.csv', index=False)
            results_indice_filtered = RocAnalysis.filter_roc_results(results_indice, min_class_spread=5)
            results_indice_filtered.to_csv(self.output_dir_roc + 'RocResults_Filtered_' + indice + '.csv', index=False)
            self.results.append(results_indice_filtered)
            self.results_temp = []
        except Exception as e:
            print("No results for " + indice + "...")
            pass
        
        def results_climo_charts(results_indice_filtered):
            df_results = results_indice_filtered.copy()
            sig_combinations = pd.concat(self.sig_combinations)
            sig_combinations.sig_stns = sig_combinations.sig_stns.apply(lambda x: ' '.join(map(str, x)) if isinstance(x, (tuple, list)) else str(x))
            for sig_id in df_results['id'].unique():
                df_id = df_results.loc[df_results['id'] == sig_id]
                fm = df_id.fuel_model.values[0]
                stns = df_id.stations.values[0]
                df_nfdrs = sig_combinations.loc[(sig_combinations.sig_stns == stns) & (sig_combinations.fuel_model == fm)]
                indice = df_id.indice.values[0]

                convert_indice = PercentileConversion(df_nfdrs, indice)
                convert_indice.make_percentile_dict()
                
                fig, ax = plt.subplots()
                plt.style.use('seaborn-v0_8-paper')
                ax.plot(df_nfdrs.groupby('doy')[indice].mean(), color='grey')
                ax.plot(df_nfdrs.groupby('doy')[indice].max(), color='red')
                ax.plot(df_nfdrs.groupby('doy')[indice].min(), color='blue')
                ax.set_ylabel('Value')
                ax.hlines(df_id.opt_value.values, 0, 365, lw=0.5, color='black')
                if indice != 'severe_fire_danger_index':
                    ax2 = ax.secondary_yaxis('right', functions=(convert_indice.value_to_percent(), convert_indice.percent_to_value()))
                    ax2.set_ylabel('Percentile')
                    plt.ylim(0)
                if indice == 'severe_fire_danger_index':
                    plt.ylim(df_nfdrs.groupby('doy')[indice].min().min())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                plt.xticks(rotation=45, ha='right')
                plt.title(self.analysis_data.fdop_name + ' ' + self.analysis_data.fdra_name + '\nIndice: ' + indice + ' & Fuel Model: ' + fm \
                    + '\nStations: ' + stns + '\n' + str(df_nfdrs.datetime.dt.year.min()) + '-' + str(df_nfdrs.datetime.dt.year.max()))
                plt.tight_layout()
                plt.savefig(self.output_dir_roc + 'Climo_' + str(df_id['id'].values[0]) + '_' + indice + '.jpg')
                plt.close('all')
        try:
            results_climo_charts(results_indice_filtered)
        except Exception as e:
            pass 
        plt.close('all')
            
    def refine_results_decision_space(self, indices):
        if os.path.isfile(self.output_dir_roc + 'RocResults_Filtered_IndiceMatches_DecisionSpace_20.csv'):
            print("Files already exist...")
        else:
            try:
                df = pd.concat(self.results, axis=0).copy()
                
                def find_ec_bi_matches(df):
                    df = df.copy()
                    df_ec_bi = df.loc[(df.indice == self.user_vars['nfdrs_names'][indices[0]]) | (df.indice == self.user_vars['nfdrs_names'][indices[1]])]
                    matching = df_ec_bi.loc[df_ec_bi.level == 4].id.duplicated()
                    matching_id = df_ec_bi.loc[df_ec_bi.level == 4].loc[matching].id.unique()
                    res = df_ec_bi.loc[df_ec_bi.id.isin(matching_id)]
                    return res
                res_matches = find_ec_bi_matches(df)
                #this is already done above but left anyway
                def remove_no_spread(df, min_class_spread):
                    rmv = df.loc[df.level > 1]
                    grp = rmv[['id', 'opt_perc_diff']].groupby(['id']).min()
                    neg = grp.loc[grp.opt_perc_diff >= min_class_spread].reset_index()
                    res = df.loc[df.id.isin(neg['id'])]        
                    return res
                    
                def refine_top_results(df, n_results):
                    min_spread = 5
                    res = df.copy()
                    while len(res.id.unique()) > n_results:
                        res = remove_no_spread(res, min_spread)
                        min_spread += 1
                    return res
                    
                r5 = refine_top_results(res_matches, 10)
                r10 = refine_top_results(res_matches, 20)
                r20 = refine_top_results(res_matches, 40)
                r5.to_csv(self.output_dir_roc + 'RocResults_Filtered_IndiceMatches_DecisionSpace_5.csv', index=False)
                r10.to_csv(self.output_dir_roc + 'RocResults_Filtered_IndiceMatches_DecisionSpace_10.csv', index=False)
                r20.to_csv(self.output_dir_roc + 'RocResults_Filtered_IndiceMatches_DecisionSpace_20.csv', index=False)
            except Exception as e:
                print("No matches exist for " + indices[0] + " and " + indices[1] + "...")


