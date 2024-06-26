######
#title: r6 transmission analysis
#version: 3.0 change code to add insitu
#version change author: maier
#author: brian maier, fs r6
#date: 02/22/2024
#note: much of this code was written on a machine circa 2010 and is inefficient due to mem errors encountered
    #may be possible to refactor code and perform more efficiently a machine with more mem
    # ex include reading/saving perimeters and zonal stats on poly with large number of fires (i.e okawen)
######

import urllib
import os
import geopandas as gpd #0.14.3
import fiona #1.9.5
import pandas as pd #2.2.0
import numpy as np #1.26.4
import gc
import rasterio
from rasterstats import zonal_stats #0.19.0
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon	

#----- begin config -----#
#create these directories prior to running
proj_dir = 'D:\\Projects\\Qwra\\Sending\\'
cwd = proj_dir + 'Powerlines\\'
perim_scratch_dir = cwd + 'Scratch\\'
#required data/input to run
ignit_dir = proj_dir + '2_1_2_Event_set\\FsimIgnitionsAll.gdb' #ignitions from FSim feature
perim_dir = proj_dir + '2_1_2_Event_set\\FsimPerimsAll.gdb' #perimeters from FSim feature
foa_dir = proj_dir + 'PNRAv2_FSim_FOAIndiv_Albers_Final.shp'
values_dir = proj_dir + 'Hvra_cNvc\\Projected\\'
#k is for file naming, v is file name
val_dict = {'PP': 'cNVC_PP.tif', 'DW': 'cNVC_DW.tif', 'INFRA': 'cNVC_INFRA.tif'}
proj_gdb = cwd + 'PLineTran_v2.gdb'
#extraneous to transmission
treemap_tif = cwd + 'treemap.tif'
treemap_dbf = cwd + 'treemap.tif.vat.dbf'
sdi_tif = cwd + 'sdi90.tif' #2023 RMAT
#get projection for project
proj_crs = gpd.read_file(ignit_dir, driver='FileGDB', layer=0).crs
#----- end config -----#

pline_seg_gdf = gpd.read_file(proj_gdb, layer='Powerlines_Pol_SplitLineAtPo_10km').to_crs(proj_crs)

#select lines that intersect fs admin boundaries
fs_admin_gdf = gpd.read_file(proj_gdb, layer='Forest_Administrative_Boundaries').to_crs(proj_crs)
pline_seg_gdf = gpd.clip(pline_seg_gdf, fs_admin_gdf)

#reset index and drop leftover cols from esri processing
pline_seg_gdf = pline_seg_gdf.reset_index(drop=True).drop(['FID', 'FID_Powerlines_Polyline_M_Buffer60', 'ORIG_FID', 'ORIG_SEQ', 'SHAPE_Length'], axis=1)

def treemap_attr(pline_gdf):
    pline_buff_gdf = pline_gdf.copy()
    pline_buff_gdf['geometry'] = pline_buff_gdf.geometry.buffer(100)
    #get treemap raster dbf file as df for raster value hash
    treemap_df = gpd.read_file(treemap_dbf).drop(['geometry'], axis=1)
    #remove treemap nodata vals
    treemap_df.replace(-99.0, np.nan, inplace=True)
    #treemap stats on 100m buffer
    zs_df = pd.DataFrame.from_dict(zonal_stats(pline_buff_gdf, treemap_tif, categorical=True, masked=True))
    #add treemap zs to lines
    pline_buff_gdf[['STANDHT_MAX', 'TPA_DEAD_MAX', 'STANDHT_MEAN', 'TPA_DEAD_MEAN']] = np.nan
    for i in zs_df.index.values:
        try:
            zs_join = zs_df.iloc[[i]].T.dropna().join(treemap_df.set_index('Value'))
            #print(i)
            pline_buff_gdf.at[i, 'STANDHT_MAX'] = zs_join.STANDHT.max()
            pline_buff_gdf.at[i, 'TPA_DEAD_MAX'] = zs_join.TPA_DEAD.max()
            pline_buff_gdf.at[i, 'STANDHT_MEAN'] = zs_join.STANDHT.mean()
            pline_buff_gdf.at[i, 'TPA_DEAD_MEAN'] = zs_join.TPA_DEAD.mean()
        except IndexError as e:
            print(e)
            print(i)
    return pline_buff_gdf.drop(['geometry'], axis=1)
pline_treemap_df = treemap_attr(pline_seg_gdf)
pline_seg_gdf = pline_seg_gdf.join(pline_treemap_df)

#remove lines with avg stand ht less than 20 feet
pline_seg_gdf = pline_seg_gdf.loc[pline_seg_gdf.STANDHT_MEAN >= 20]
#remove lines with no treedata 
pline_seg_gdf = pline_seg_gdf.dropna(subset=['STANDHT_MEAN'], axis=0)

#get foa that overlay with lines
def foa_overlay(pline_gdf):
    foa_gdf = gpd.read_file(foa_dir).to_crs(proj_crs)
    pline_gdf['geometry'] = pline_gdf.geometry.buffer(100)
    gdf = gpd.overlay(foa_gdf, pline_gdf, how='intersection')
    return gdf.dissolve(by='FOA_number').reset_index()[['FOA_number', 'geometry']]
foas_gdf = foa_overlay(pline_seg_gdf)

#reduce processing of points in ignition gdb by foa
foas_gdf['FOA_number'] = foas_gdf['FOA_number'].astype(str)
foa_interest = foas_gdf['FOA_number'].values
ignition_layers = fiona.listlayers(ignit_dir)
perimeter_layers = fiona.listlayers(perim_dir)
def foa_select(foa_list): 
	l = []
	for foa in foa_list:
		for foa_num in foa_interest:
			if foa_num in foa[4:7]:
				l.append(foa)
	return l
foa_ignitions = foa_select(ignition_layers)
foa_perimeters = foa_select(perimeter_layers)

#create analysis polys and analysis area
pline_buff_gdf = pline_seg_gdf.copy()
pline_buff_gdf['geometry'] = pline_buff_gdf.geometry.buffer(400)
analysis_gdf = pline_buff_gdf.dissolve()
analysis_gdf = gpd.GeoDataFrame(analysis_gdf.reset_index(drop=True), geometry=analysis_gdf.geometry)

#perform overlay of ignitions with analysis area, dump those outside analysis area and merge
def ignition_sjoin(analysis_gdf):
	print("Performing spatial join of ignitions with analysis area...")
	l = []
	for foa in foa_ignitions:
		print('Reading...' + foa)
		gdf = gpd.read_file(ignit_dir, driver='FileGDB', layer=foa).to_crs(proj_crs)
		gdf.reset_index(drop=True, inplace=True)
		gdf_clip = gpd.overlay(gdf, analysis_gdf)
		gdf_clip['FOA'] = foa[4:7]
		if gdf_clip.empty:
			print("Empty FOA...")
		else:
			l.append(gdf_clip)
			#print(gdf_clip)
	gdf_merge = pd.concat(l, ignore_index=True)
	return gdf_merge.to_crs(proj_crs)
ignitions_gdf = ignition_sjoin(analysis_gdf)

ignitions_gdf['FIRE'] = ignitions_gdf['FIRE_NUMBE'].astype(int)
#create unique id for attribute join to perimeters	
ignitions_gdf['FOA_FIRE'] = ignitions_gdf['FOA'].astype(str) + '_' + \
	ignitions_gdf['FIRE'].astype(str)
ignitions_gdf = ignitions_gdf[['FOA_FIRE', 'geometry']]		
ignitions_list = ignitions_gdf['FOA_FIRE'].values
ignitions_gdf.to_file(proj_gdb, layer='Ignits_From', driver='OpenFileGDB')

#get rid of any trash from previous run if needed
def clean_scratch(folder):
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
clean_scratch(cwd + 'Scratch')

#select perimeters which originated in the analysis area by matching with ignitions_list
def perimeter_select():
	print("Selecting perimeters that match analysis area ignitions...")
	l = []
	for foa in foa_perimeters:
		print('Reading...' + foa)
		gdf = gpd.read_file(perim_dir, driver='FileGDB', layer=foa).drop(['Shape_Length', 'Shape_Area'], axis=1)
		gdf['FIRE'] = gdf['FIRE_NUMBE'].astype(int)
		gdf['FOA'] = foa[4:7]
		gdf['FOA_FIRE'] = gdf['FOA'].astype(str) + '_' + gdf['FIRE'].astype(str)
		ignitions_mask = gdf['FOA_FIRE'].isin(ignitions_list) #boolean df
		gdf = gdf[ignitions_mask]
		if gdf.empty:
			print("Empty FOA...")
		else:
			#print(gdf)
			print("Writing Perimeters...")
			try:
				fname = foa + '.shp'
				print("Writing..." + fname)
				gdf.to_file(perim_scratch_dir + fname, driver='ESRI Shapefile')
			except ValueError:
				print("Empty DataFrame...")
				pass
			l.append(gdf)
			del [[gdf, ignitions_mask]]
			print("Trash man...")
			gc.collect()
	return l
perim_selected = perimeter_select()
perims_merged = pd.concat(perim_selected, ignore_index=True)
perims_merged.to_file(proj_gdb, layer='Perims_From', driver='OpenFileGDB')

#required to save geojson (gdf) to gdb (single dtype only)
def single_multi(gdf):
	gdf['geometry'] = [MultiPolygon([feature]) if isinstance(feature, Polygon) \
		else feature for feature in gdf['geometry']]
	return gdf
#scale the numerical values in the dataframe to be between -1 and 1, preserving the signal of all values	
def zero_centered_min_max_scaling(df, column):
	max_absolute_value = df[column].abs().max()
	df[column] = df[column] / max_absolute_value
	return df

#not needed for pod feature which already has uniqueid field
pline_buff_gdf.loc[:, 'UniqueID'] = pline_buff_gdf.index

perims_merged = gpd.read_file(proj_gdb, layer='Perims_From').to_crs(proj_crs)
ignitions_gdf = gpd.read_file(proj_gdb, layer='Ignits_From').to_crs(proj_crs)

#new version to include in situ vs source risk and refactor old code
#iterating through indiv polys as some analysis poly foa perims feature are too large to read causing mem error
l = []
for i_analysis_poly, row_analysis_poly in pline_buff_gdf.iterrows():
    row_analysis_poly = gpd.GeoDataFrame(pd.DataFrame(row_analysis_poly).T).set_crs(proj_crs)
    poly_ignitions = gpd.overlay(ignitions_gdf, row_analysis_poly)
    poly_ignitions_list = poly_ignitions['FOA_FIRE'].values
    poly_perims_mask = perims_merged['FOA_FIRE'].isin(poly_ignitions_list)
    poly_perims = perims_merged[poly_perims_mask]
    for i_perim, row_perim in poly_perims.iterrows():
        row_perim_gdf = gpd.GeoDataFrame(pd.DataFrame(row_perim).T).set_crs(proj_crs)
        insitu_perim_gdf = gpd.clip(row_perim_gdf, row_analysis_poly)
        for k, v in val_dict.items():
            hvra_dir = values_dir + v
            zs_source_res_df = pd.DataFrame.from_dict(zonal_stats(row_perim_gdf, hvra_dir, stats='sum'))
            zs_source_res_df = zs_source_res_df.rename({'sum': 'source_sum'}, axis=1)
            zs_insitu_res_df = pd.DataFrame.from_dict(zonal_stats(insitu_perim_gdf, hvra_dir, stats='sum'))
            zs_insitu_res_df = zs_insitu_res_df.rename({'sum': 'insitu_sum'}, axis=1)
            zs_res = row_perim_gdf.reset_index(drop=True).join(zs_source_res_df)
            zs_res = zs_res.join(zs_insitu_res_df[['insitu_sum']])
            zs_res['Hvra'] = k
            zs_res['UniqueID'] = row_analysis_poly.index[0]
            if not zs_res.empty:
                l.append(zs_res)
                #print(zs_res['FOA_FIRE'])
    #print(i_analysis_poly)

#merge individual perims back into single feature
zs_perims = pd.concat(l, ignore_index=True)
zs_perims.reset_index(drop=True, inplace=True)
zs_perims.to_file(proj_gdb, layer='ZsPerims', driver='OpenFileGDB')

#seperate merged perim feature by hvra, save perims and ignits
for k, v in val_dict.items():
    zs_val_perims = zs_perims.loc[zs_perims.Hvra == k]
    zs_val_perims.to_file(proj_gdb, layer='ZsPerims_' + k, driver='OpenFileGDB')
    perims_sent_l = zs_val_perims['FOA_FIRE']
    ignitions_sent_mask = ignitions_gdf['FOA_FIRE'].isin(perims_sent_l)
    ignitions_sent = ignitions_gdf[ignitions_sent_mask]
    ignitions_sent = ignitions_sent.merge(zs_val_perims.drop('geometry', axis=1))
    ignitions_sent.to_file(proj_gdb, layer='ZsIgnits_' + k, driver='OpenFileGDB')

#calculate tranmission
for k, v in val_dict.items():
    print(k)
    #get count of ignitions sent for normalization (suggested by J from Pyrologix as opposed to acres circa 2015)
    ignitions_sent_gdf = gpd.read_file(proj_gdb, layer='ZsIgnits_' + k).to_crs(proj_crs)
    ignitions_sent_gdf.reset_index(drop=True, inplace=True)
    ignitions_all = gpd.sjoin(pline_buff_gdf[['UniqueID', 'geometry']], ignitions_gdf)[['UniqueID']]
    ignitions_all['count_all'] = 1
    ignitions_all_count = ignitions_all.groupby('UniqueID').sum()

    def join_stats_analysis_poly():
        gdf = ignitions_sent_gdf
        print("Joining perimeter zonal statistics sum to ignitions...")
        gdf_sjoin = gpd.sjoin(gdf, pline_buff_gdf.reset_index()[['UniqueID', 'geometry']], \
              how='left', predicate='intersects')#[['FOA_FIRE', 'source_sum', 'insitu_sum', 'NUM_ITR', 'UniqueID', 'geometry']] #.rename({'UniqueID_l':'UniqueID'}, axis=1)
        print("Writing joined shapefile...")
        gdf_sjoin = single_multi(gdf_sjoin)
        gdf_sjoin.to_file(proj_gdb, layer='IgnitsStats_' + k, driver='OpenFileGDB')
    join_stats_analysis_poly()	

    def transmission_calc():	
        print('Final huc sending to wpl calculations...')
        gdf = gpd.read_file(proj_gdb, layer='ZsIgnits_' + k).to_crs(proj_crs)
        gdf = gdf.loc[:, ['UniqueID', 'source_sum', 'insitu_sum', 'NUM_ITR']]
        gdf['count'] = 1
        gdf['source_sum_iter'] = gdf['source_sum'] / gdf['NUM_ITR'].astype('float')
        gdf['insitu_sum_iter'] = gdf['insitu_sum'] / gdf['NUM_ITR'].astype('float')
        group = gdf.groupby(by='UniqueID').sum()
        group = group.join(ignitions_all_count)
        group['source_sum_rs'] = (group['source_sum_iter'] / group['count_all']) * 10000 #number of iterations from some, not all, foa
        group['insitu_sum_rs'] = (group['insitu_sum_iter'] / group['count_all']) * 10000

        group.reset_index(inplace=True)
        group = group[['UniqueID', 'source_sum_rs', 'insitu_sum_rs', 'count_all']]
        gdf_results = pline_buff_gdf.reset_index().merge(group, on='UniqueID', how='left')
        gdf_results['source_sum_rs'] = gdf_results['source_sum_rs'].fillna(0)		
        gdf_results['insitu_sum_rs'] = gdf_results['insitu_sum_rs'].fillna(0)       
        zero_centered_min_max_scaling(gdf_results, 'source_sum_rs')
        zero_centered_min_max_scaling(gdf_results, 'insitu_sum_rs')        
        cols = ['UniqueID', 'source_sum_rs', 'insitu_sum_rs', 'count_all', 'geometry']
        gdf_results = gdf_results[cols]
        gdf_results = single_multi(gdf_results)
        gdf_results.to_file(proj_gdb, layer='Transmission_' + k, driver='OpenFileGDB')
    transmission_calc()	

def combine_results():
    l=[]
    for k, v in val_dict.items():
        gdf = gpd.read_file(proj_gdb, layer='Transmission_' + k).to_crs(proj_crs)
        gdf.rename({'source_sum_rs': k + '_SourceRisk', 'insitu_sum_rs': k + '_InSituRisk'}, axis=1, inplace=True)
        gdf.drop(['count_all', 'geometry'], axis=1, inplace=True)
        gdf.set_index('UniqueID', inplace=True)
        l.append(gdf)
    return l

res = pd.concat(combine_results(), axis=1)
res_gdf = pline_buff_gdf.set_index('UniqueID')[['geometry']].join(res)
single_multi(res_gdf).to_file(proj_gdb, layer='Transmission_Combined', driver='OpenFileGDB')

#do some stuff with the tree layer for a probability metric
res_tree = gpd.read_file(proj_gdb, layer='Transmission_Combined').to_crs(proj_crs)
res_tree = res_tree.set_index('UniqueID').join(pline_buff_gdf.set_index('UniqueID').drop(['geometry'], axis=1))
res_tree['tree_mult'] = res_tree.STANDHT_MEAN * res_tree.TPA_DEAD_MEAN
zero_centered_min_max_scaling(res_tree, 'tree_mult')

cols = ['tree_mult','PP_SourceRisk','PP_InSituRisk','DW_SourceRisk','DW_InSituRisk','INFRA_SourceRisk','INFRA_InSituRisk','geometry']
res_tree = res_tree[cols]
res_tree_mult = res_tree[['PP_SourceRisk','PP_InSituRisk','DW_SourceRisk','DW_InSituRisk','INFRA_SourceRisk','INFRA_InSituRisk']] \
    .apply(lambda x: x * res_tree.tree_mult)

for col in res_tree_mult:
    zero_centered_min_max_scaling(res_tree_mult, col)
    
single_multi(pline_buff_gdf.set_index('UniqueID')[['geometry']].join(res_tree_mult)).to_file(proj_gdb, layer='Transmission_Tree', driver='OpenFileGDB')

#add sdi as attr to results
sdi_df = pd.DataFrame.from_dict(zonal_stats(pline_buff_gdf, sdi_tif, stats='mean'))
sdi_df.rename({'mean': 'Sdi90_Mean'}, axis=1, inplace=True)

res_sdi = gpd.read_file(proj_gdb, layer='Transmission_Combined').to_crs(proj_crs)
res_sdi.join(sdi_df).to_file(proj_gdb, layer='Transmission_Combined_Sdi90', driver='OpenFileGDB')

res_sdi_tree = gpd.read_file(proj_gdb, layer='Transmission_Tree').to_crs(proj_crs)
res_sdi.join(sdi_df).to_file(proj_gdb, layer='Transmission_Tree_Sdi90', driver='OpenFileGDB')

