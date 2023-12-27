# -*- coding: utf-8 -*-
"""
Extract metadata for PANGAEA from narrowband SIMRAD echosounder data using echopype
Created on Wed Aug 16 12:50:47 2023
@author: Roland Proud rp43@st-andrews.ac.uk

Metadata column names and definitions (note values are case sensitive)

Event	                                            [string] cruise label
Echogram, raw format []	                            [string] Name of raw data file (e.g. D20180718-T020310.raw)
Identification []//*channel	                        [string] (e.g. GPT  18 kHz 009072056150 1-1 ES18-11)
Data acquisition software (recording raw data) []	[string] software + version (e.g. ER60 v1.0)
Frequency [kHz]	                                    [numeric] (e.g. 18.0)
Number of pings []	                                [integer] (e.g. 1000)
Beamwidth, alongship [deg]	                        [numeric] (e.g. 7.0) also known as minor angle. Calibration parameter.
Beamwidth, athwartship [deg]	                        [numeric] (e.g. 7.0) also known as major angle. Calibration parameter.
Gain, transducer [dB re 1]	                        [numeric] (e.g 24.5). Calibration parameter.
Simrad correction factor [dB re 1/m]	                [numeric] (e.g 0.1). Calibration parameter.
Sample interval [s]	                                [numeric] (e.g. 0.0001)
Pulse duration, transmitted [ms]	                    [numeric] (e.g. 1.024)
Power, transmitted [W]	                            [numeric] (e.g. 2000)
Sound velocity in water [m/s]	                    [numeric] (e.g. 1500)
Sound absorption [dB/m]	                            [numeric] (e.g. 0.003)
Depth, water, top/minimum [m]	                    [numeric] (e.g. 3) typically transducer depth
Depth, water, bottom/maximum [m]	                    [numeric] (e.g. 1000) typically max recording depth
DEPTH, water [m]	                                    [numeric] (e.g. 1000) Calculated median between min and max.
Date/time start []	                                [string] (e.g. 01/01/2023 01:00:00.123) If milliseconds are given these must have exactly three decimals.
Date/time end []	                                    [string] (e.g. 01/01/2023 01:00:00.123) If milliseconds are given these must have exactly three decimals.
Latitude []	                                        [numeric] (e.g. 5.0) start latitude
Latitude 2 []	                                    [numeric] (e.g. 5.0) end latitude
Longitude []	                                        [numeric] (e.g. 5.0) start longitude
Longitude 2 []	                                    [numeric] (e.g. 5.0) end longitude
Calibration []	                                    [string factor: Yes/No/Unknown] (e.g. Yes)

"""

# imports
import os
import glob
import numpy as np
import echopype as ep ## version 0.7.1
import pandas as pd  
from collections import defaultdict

# paras
path_to_data       = 'path_to_raw_files'
cruise_label       = "SURVEY_YEAR" # Cruise label (ask PANGAEA)
ref_ping           = 0          # ping number for extraction of information [don't change]
calibration_status = "Unknown"  # change to Yes or No if known

# change working directory
os.chdir(path_to_data)

# list file names
file_names = np.sort(glob.glob("*.raw"))

# dictionary to hold metadata
metadata = defaultdict(list)

# file index
for file_idx in range(len(file_names)):
    print("processing: " + file_names[file_idx])

    # read file (only CW data recorded using either EK60 or EK80)
    raw_type = 'EK60'
    try:
        ek = ep.open_raw(file_names[file_idx], sonar_model= raw_type)
    except:
        raw_type = 'EK80'
        ek       = ep.open_raw(file_names[file_idx], sonar_model= raw_type)
    
    # freqs and channels
    all_frequencies = ek['Sonar/Beam_group1']['frequency_nominal'].data
    all_channels    = ek['Sonar/Beam_group1'].coords['channel'].data
    
    ## calculate Sv (main data - backscattering intensity)
    try:
        Sv_obj = ep.calibrate.compute_Sv(ek,waveform_mode = "CW", encode_mode = "power") 
        print('Power and angle data used to calculate Sv...')
    except:
        Sv_obj = ep.calibrate.compute_Sv(ek,waveform_mode = "CW", encode_mode = "complex")
        print('Complex data used to calculate Sv...')
                
    # metadata
    software              = ek['Sonar'].sonar_software_name + ' v' +ek['Sonar'].sonar_software_version
    npings                = len(Sv_obj['Sv'].coords['ping_time'])
    max_recording_range   = np.round(np.nanmax(Sv_obj['echo_range'].data),5)
    
    # get GPS info
    gps            = {}
    sentence_types = np.array(ek['Platform']['sentence_type'].data)
    all_lats       = np.array(ek['Platform']['latitude'].data)
    all_lons       = np.array(ek['Platform']['longitude'].data)
    all_times      = np.array(ek['Platform']['time1'].data) ## time1 - GPS fixes
    for st in np.unique(sentence_types):
        ## skip invalid sentence types
        if st not in ['GGA','GLL','RMC']:
            continue
        gps[st] = {}
        idx     = np.where(sentence_types == st)[0]
        if len(idx) > 0:
            gps['time']      = all_times[idx]
            gps['latitude']  = all_lats[idx]
            gps['longitude'] = all_lons[idx]
            break
        
    ## gets start/end fixes
    gps_start_dt = gps['time'][0]
    gps_end_dt   = gps['time'][-1]
    
    ## for positions, take median over first few and last fixes
    value_range   = min(10,len(gps['time'])/2 -1)
    value_range   = max(value_range,1)
    gps_start_lat = np.round(np.nanmedian(gps['latitude'][:value_range]),5)
    gps_end_lat   = np.round(np.nanmedian(gps['latitude'][-value_range:]),5)
    gps_start_lon = np.round(np.nanmedian(gps['longitude'][:value_range]),5)
    gps_end_lon   = np.round(np.nanmedian(gps['longitude'][-value_range:]),5)
              
    # metadata by channel
    for i in range(len(all_frequencies)):
        # by channel metadata
        pulse_duration        = ek['Sonar/Beam_group1']['transmit_duration_nominal'].loc[dict(channel=all_channels[i])].data[ref_ping]
        pulse_duration        = round(pulse_duration,6)
        sample_interval       = ek['Sonar/Beam_group1']['sample_interval'].loc[dict(channel=all_channels[i])].data[ref_ping]
        sample_interval       = round(sample_interval,7)
        transmit_power        = ek['Sonar/Beam_group1']['transmit_power'].loc[dict(channel=all_channels[i])].data[ref_ping]
        
        # CHECK
        try:
            beamwidth_alongship   = np.round(float(ek['Sonar/Beam_group1']['beamwidth_twoway_alongship'].loc[dict(channel=all_channels[i])].data[ref_ping]),5)
            beamwidth_athwartship = np.round(float(ek['Sonar/Beam_group1']['beamwidth_twoway_athwartship'].loc[dict(channel=all_channels[i])].data[ref_ping]),5)
        except:
            beamwidth_alongship   = Sv_obj['beamwidth_alongship'].loc[dict(channel=all_channels[i])].data[ref_ping]
            beamwidth_athwartship = Sv_obj['beamwidth_athwartship'].loc[dict(channel=all_channels[i])].data[ref_ping]
        ## calibration
        gain                  = Sv_obj['gain_correction'].loc[dict(channel=all_channels[i])].data[ref_ping]
        Sa                    = Sv_obj['sa_correction'].loc[dict(channel=all_channels[i])].data[ref_ping]
        # CHECK
        try:
            sound_speed           = np.round(float(Sv_obj['sound_speed'].data),5)
            sound_absorption      = np.round(float(Sv_obj['sound_absorption'].data),8)
            transducer_depth      = np.round(Sv_obj['water_level'].data[0],5) ## CHECK
        except:
            sound_speed           = np.round(Sv_obj['sound_speed'].loc[dict(channel=all_channels[i])].data[ref_ping],5)
            sound_absorption      = np.round(Sv_obj['sound_absorption'].loc[dict(channel=all_channels[i])].data[ref_ping],8)
            transducer_depth      = np.round(float(Sv_obj['water_level'].loc[dict(channel=all_channels[i])].data[ref_ping]),5) ## CHECK
    
        # add to dictionary
        metadata['Event'].append(cruise_label)
        metadata['Echogram, raw format []'].append(file_names[file_idx])
        metadata['Identification []//*channel'].append(all_channels[i])
        metadata['Data acquisition software (recording raw data) []'].append(software)
        metadata['Frequency [kHz]'].append(all_frequencies[i]/1000)
        metadata['Number of pings []'].append(npings)
        metadata['Beamwidth, alongship [deg]'].append(beamwidth_alongship)
        metadata['Beamwidth, athwartship [deg]'].append(beamwidth_athwartship)
        metadata['Gain, transducer [dB re 1]'].append(gain)
        metadata['Simrad correction factor [dB re 1/m]'].append(Sa)
        metadata['Sample interval [s]'].append(sample_interval)
        metadata['Pulse duration, transmitted [ms]'].append(round(pulse_duration*1000,3))
        metadata['Power, transmitted [W]'].append(transmit_power)
        metadata['Sound velocity in water [m/s]'].append(sound_speed)
        metadata['Sound absorption [dB/m]'].append(sound_absorption)
        metadata['Depth, water, top/minimum [m]'].append(transducer_depth)
        metadata['Depth, water, bottom/maximum [m]'].append(max_recording_range + transducer_depth)
        metadata['DEPTH, water [m]'].append(transducer_depth + max_recording_range/2)
        metadata['Date/time start []'].append(gps_start_dt)
        metadata['Date/time end []	'].append(gps_end_dt)
        metadata['Latitude []'].append(gps_start_lat)
        metadata['Latitude 2 []'].append(gps_end_lat)
        metadata['Longitude []	'].append(gps_start_lon)
        metadata['Longitude 2 []'].append(gps_end_lon)
        metadata['Calibration []'].append(calibration_status)
    
    
## to dataframe
pd.options.display.float_format = '{:.0f}'.format
df = pd.DataFrame.from_dict(metadata)

## to file
df.to_csv(cruise_label + "_metadata.csv", index=False,  float_format= '%f', date_format='%d/%m/%Y %H:%M:%S')




