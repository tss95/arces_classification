"""
Beam waveform extraction utilities.

Isolated module for loading and creating beam waveforms from seismic array data.
To be integrated into seismonpy
"""

import numpy as np


class EventWindow():
    def __init__(self, station=None, window_start=None, window_end=None, baz=None, p_beam_vel=None, s_beam_vel=None,
                 lower_frequency=None, upper_frequency=None, taper=0.05):
        """
        Initialize EventWindow object
        :param station: Station Name
        :type station: str
        :param window_start: start time of window
        :type window_start: timestamp
        :param window_end: end time of window
        :type window_end: timestamp                  
        :param baz: Back-azimuth of event at station
        :type baz: float
        :param p_beam_vel: P wave velocity for beamforming (km/s)
        :type p_beam_vel: float
        :param s_beam_vel: S wave velocity for beamforming (km/s)
        :type s_beam_vel: float
        :param lower_frequency: Lower frequency for bandpass filter (Hz)
        :type lower_frequency: float
        :param upper_frequency: Upper frequency for bandpass filter (Hz)
        :type upper_frequency: float
        :param taper: Taper fraction for preprocessing (default 0.05)
        :type taper: float
        """
        self.station = station
        self.window_start = window_start
        self.window_end = window_end
        self.baz = baz
        self.p_beam_vel = p_beam_vel
        self.s_beam_vel = s_beam_vel
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.taper = taper
from obspy import UTCDateTime
from seismonpy.core import SeismonStream


def get_array_info():
    """
    Return a dictionary containing information about seismic arrays.
    
    For each array, the dictionary contains:
    - Longitude/Latitude: Array location
    - Stations: List of station codes belonging to the array
    - Time Periods: List of time periods divided by major upgrades
    - Channel Names: Channel base names for data available in each time period
    - Three Component Stations: Stations with 3C data in each time period ('all' if all stations)
    - Reference Time: Reference time with full array recording (needed for ArrayNet)
    
    :return: Dictionary with array information
    :rtype: dict
    """
    array_stations = {}
    array_stations['ARCES'] = {}
    array_stations['ARCES']['Longitude'] = 25.5058
    array_stations['ARCES']['Latitude'] = 69.5349
    # Which stations belong to array
    array_stations['ARCES']['Stations'] = ['ARA0', 'ARA1', 'ARA2', 'ARA3', 'ARB1', 'ARB2', 'ARB3',
                           'ARB4', 'ARB5', 'ARC1', 'ARC2', 'ARC3', 'ARC4', 'ARC5',
                           'ARC6', 'ARC7', 'ARD1', 'ARD2', 'ARD3', 'ARD4', 'ARD5',
                           'ARD6', 'ARD7', 'ARD8', 'ARD9', 'ARE0']
    # Time periods devided by major upgrades
    array_stations['ARCES']['Time Periods'] = [['1987-08-30T00:00:00', '2014-09-17T12:00:00'],
                                           ['2014-09-17T12:00:00', '-']]
    # Channel base names for data available in these time periods 
    array_stations['ARCES']['Channel Names'] = [['s'], ['BH', 'HH']]
    # Which stations have three-component data in these time periods
    array_stations['ARCES']['Three Component Stations'] = [['ARA0', 'ARC2', 'ARC4', 'ARC7'], ['all']]
    # Reference time with full array recording (needed for ArrayNet)
    array_stations['ARCES']['Reference Time'] = '2017-04-22T00:18:30'

    array_stations['FINES'] = {}
    array_stations['FINES']['Longitude'] = 26.0771
    array_stations['FINES']['Latitude'] = 61.4436
    array_stations['FINES']['Stations'] = ['FIA0', 'FIA1', 'FIA2', 'FIA3', 'FIB1', 'FIB2', 'FIB3',
                           'FIB4', 'FIB5', 'FIB6', 'FIC1', 'FIC2', 'FIC3', 'FIC4',
                           'FIC5', 'FIC6']
    array_stations['FINES']['Time Periods'] = [['1993-11-18T00:00:00', '2007-07-11T00:00:00'],
                                           ['2007-07-11T00:00:00', '2023-09-12T07:00:00'],
                                           ['2023-09-12T07:00:00', '-']]
    array_stations['FINES']['Channel Names'] = [['s', 'b'], ['SH', 'BH', 'HH'], ['HH']]
    array_stations['FINES']['Three Component Stations'] = [['FIA0', 'FIA1'], ['FIA0', 'FIA1'], ['all']]
    array_stations['FINES']['Reference Time'] = '2022-01-01T00:30:00'

    array_stations['NORES'] = {}
    array_stations['NORES']['Longitude'] = 11.5414
    array_stations['NORES']['Latitude'] = 60.7353
    array_stations['NORES']['Stations'] = ['NRA0', 'NRA1', 'NRA2', 'NRA3', 'NRB1', 'NRB2', 'NRB3',
                           'NRB4', 'NRB5', 'NRC1', 'NRC2', 'NRC3', 'NRC4', 'NRC5',
                           'NRC6', 'NRC7', 'NRD1', 'NRD2', 'NRD3', 'NRD4', 'NRD5',
                           'NRD6', 'NRD7', 'NRD8', 'NRD9', 'NRE0']
    array_stations['NORES']['Time Periods'] = [['1984-10-03T15:00:00', '2002-06-11T11:59:00'],
                                           ['2010-12-20T12:00:00', '2015-08-13T10:00:00'],
                                           ['2015-08-13T10:00:00', '-']]
    array_stations['NORES']['Channel Names'] = [['s'], ['SH'], ['HH']]
    array_stations['NORES']['Three Component Stations'] = [['NRA0', 'NRC2', 'NRC4', 'NRC7'], ['all'], ['all']]
    array_stations['NORES']['Reference Time'] = '2020-03-01T10:42:00'

    array_stations['SPITS'] = {}
    array_stations['SPITS']['Longitude'] = 16.3700
    array_stations['SPITS']['Latitude'] = 78.1777
    array_stations['SPITS']['Stations'] = ['SPA0', 'SPA1', 'SPA2', 'SPA3', 'SPB1', 'SPB2', 'SPB3',
                           'SPB4', 'SPB5']
    array_stations['SPITS']['Time Periods'] = [['1992-11-06T00:00:00', '2004-08-12T23:59:59'],
                                           ['2004-08-13T00:00:00', '2015-05-05T15:00:00'],
                                           ['2015-05-07T13:00:00', '-']]
    array_stations['SPITS']['Channel Names'] = [['s', 'b'], ['BH'], ['HH']]
    array_stations['SPITS']['Three Component Stations'] = [['SPB4'], ['SPA0', 'SPB1', 'SPB2', 'SPB3', 'SPB4', 'SPB5'],
                                                       ['SPA0', 'SPB1', 'SPB2', 'SPB3', 'SPB4', 'SPB5']]
    array_stations['SPITS']['Reference Time'] = '2019-03-13T22:28:00'

    array_stations['HNAR'] = {}
    array_stations['HNAR']['Stations'] = ['HNA0', 'HNA1', 'HNA2', 'HNA3', 'HNB1', 'HNB2', 'HNB3',
                          'HNB4', 'HNB5']
    array_stations['HNAR']['Longitude'] = 4.9571
    array_stations['HNAR']['Latitude'] = 60.6106
    array_stations['HNAR']['Time Periods'] = [['2020-05-06T00:00:00', '-']]
    array_stations['HNAR']['Channel Names'] = [['HH']]
    array_stations['HNAR']['Three Component Stations'] = [['all']]
    array_stations['HNAR']['Reference Time'] = '2023-05-01T00:00:00'

    array_stations['HSPA'] = {}
    array_stations['HSPA']['Stations'] = ['HSPA1', 'HSPA2', 'HSPA3', 'HSPA4', 'HSPA5']
    array_stations['HSPA']['Longitude'] = 15.535683
    array_stations['HSPA']['Latitude'] = 77.003525
    array_stations['HSPA']['Time Periods'] = [['2022-06-21T00:00:00', '-']]
    array_stations['HSPA']['Channel Names'] = [['HH']]
    array_stations['HSPA']['Three Component Stations'] = [['all']]
    array_stations['HSPA']['Reference Time'] = '2023-05-01T00:00:00'

    array_stations['HFS'] = {}
    array_stations['HFS']['Stations'] = ['HFSA1', 'HFSB1', 'HFSB2', 'HFSB3', 'HFSB4', 'HFSB5', 'HFSC1',
                          'HFSC2', 'HFC2', 'HFA0', 'HFA1', 'HFA2', 'HFA3', 'HFB1', 'HFB2', 'HFB3', 'HFB4', 'HFB5']
    array_stations['HFS']['Longitude'] = 13.6945
    array_stations['HFS']['Latitude'] = 60.1335
    array_stations['HFS']['Time Periods'] = [['1992-01-01T:00:00:00', '2001-08-24T00:00:00'],
                                             ['2001-08-24T00:00:00', '2022-01-20T00:00:00'],
                                             ['2022-01-20T00:00:00', '-']]
    array_stations['HFS']['Channel Names'] = [['s', 'b'], ['s', 'b'], ['HH']]
    array_stations['HFS']['Three Component Stations'] = [['HFSC2'], ['HFC2'], ['all']]
    array_stations['HFS']['Reference Time'] = '2023-05-01T00:00:00'

    array_stations['NOA'] = {}
    array_stations['NOA']['Stations'] = ['NB200', 'NB201', 'NB202', 'NB203', 'NB204', 'NB205']
    array_stations['NOA']['Longitude'] = 11.2148
    array_stations['NOA']['Latitude'] = 61.0397
    array_stations['NOA']['Time Periods'] = [['1995-01-01T00:00:00', '2012-06-20T00:00:00'],
                                             ['2012-06-20T00:00:00', '-']]
    array_stations['NOA']['Channel Names'] = [['s', 'b'], ['BH']]
    array_stations['NOA']['Three Component Stations'] = [['NB201'], ['NB201']]
    array_stations['NOA']['Reference Time'] = '2023-05-01T00:00:00'

    # used by AFTAC
    array_stations['NB2'] = {}
    array_stations['NB2']['Stations'] = ['NB200', 'NB201', 'NB202', 'NB203', 'NB204', 'NB205']
    array_stations['NB2']['Longitude'] = 11.2148
    array_stations['NB2']['Latitude'] = 61.0397
    array_stations['NB2']['Time Periods'] = [['1995-01-01T00:00:00', '2012-06-20T00:00:00'],
                                             ['2012-06-20T00:00:00', '-']]
    array_stations['NB2']['Channel Names'] = [['s', 'b'], ['BH']]
    array_stations['NB2']['Three Component Stations'] = [['NB201'], ['NB201']]
    array_stations['NB2']['Reference Time'] = '2023-05-01T00:00:00'

    # only use one time period for following arrays
    array_stations['TXAR'] = {}
    array_stations['TXAR']['Stations'] = ['TX01', 'TX02', 'TX03', 'TX04', 'TX05', 'TX06', 'TX07', 'TX08', 'TX09', 'TX10', 'TX31']
    array_stations['TXAR']['Longitude'] = -103.6677
    array_stations['TXAR']['Latitude'] = 29.334289
    array_stations['TXAR']['Time Periods'] = [['1994-01-01T00:00:00', '-']]
    array_stations['TXAR']['Channel Names'] = [['s', 'b', 'BH', 'SH']]
    array_stations['TXAR']['Three Component Stations'] = [['all']]
    array_stations['TXAR']['Reference Time'] = '2023-05-01T00:00:00'

    array_stations['MKAR'] = {}
    array_stations['MKAR']['Stations'] = ['MK01', 'MK02', 'MK03', 'MK04', 'MK05', 'MK06', 'MK07', 'MK08', 'MK09', 'MK31', 'MK32']
    array_stations['MKAR']['Longitude'] = 82.2904
    array_stations['MKAR']['Latitude'] = 46.7937
    array_stations['MKAR']['Time Periods'] = [['2000-11-14T00:00:00', '-']]
    array_stations['MKAR']['Channel Names'] = [['BH', 'SH']]
    array_stations['MKAR']['Three Component Stations'] = [['MK32']]
    array_stations['MKAR']['Reference Time'] = '2023-05-01T00:00:00'

    array_stations['BVAR'] = {}
    array_stations['BVAR']['Stations'] = ['BVA0', 'BVA1', 'BVA2', 'BVA3', 'BVA4', 'BVB5', 'BVB6', 'BVB7', 'BVB8', 'BVB9']
    array_stations['BVAR']['Longitude'] = 70.3885
    array_stations['BVAR']['Latitude'] = 53.0249
    array_stations['BVAR']['Time Periods'] = [['2002-07-06T00:00:00', '-']]
    array_stations['BVAR']['Channel Names'] = [['BH', 'SH']]
    array_stations['BVAR']['Three Component Stations'] = [['BVA0']]
    array_stations['BVAR']['Reference Time'] = '2023-05-01T00:00:00'

    array_stations['GERES'] = {}
    array_stations['GERES']['Stations'] = ['GEA0', 'GEA1', 'GEA2', 'GEA3', 'GEB1', 'GEB2', 'GEB3',
                           'GEB4', 'GEB5', 'GEC1', 'GEC2', 'GEC2A', 'GEC2B', 'GEC3', 'GEC4', 'GEC5',
                           'GEC6', 'GEC7', 'GEC7A', 'GED1', 'GED2', 'GED3', 'GED4', 'GED5',
                           'GED6', 'GED7', 'GED8', 'GED9']
    array_stations['GERES']['Longitude'] = 48.8368
    array_stations['GERES']['Latitude'] = 13.7019
    array_stations['GERES']['Time Periods'] = [['2000-01-01T00:00:00', '-']]
    array_stations['GERES']['Channel Names'] = [['HH', 'SH']]
    array_stations['GERES']['Three Component Stations'] = [['GEA2', 'GEC2', 'GEC2A', 'GEC7A', 'GED1', 'GED4', 'GED7']]
    array_stations['GERES']['Reference Time'] = '2023-05-01T00:00:00'

    return array_stations


def adjust_trace_length(st, target_length, sample_tolerance=10):
    """
    Adjust trace lengths to target length by trimming or zero-padding.
    
    :param st: Stream with traces to adjust
    :type st: SeismonPy Stream object
    :param target_length: Target number of samples
    :type target_length: int
    :param sample_tolerance: Tolerance in samples for length differences
    :type sample_tolerance: int
    :return: Adjusted stream, or None if tolerance exceeded
    :rtype: SeismonPy Stream object or None
    """
    test = list(set([tr.stats.npts for tr in st]))
    if len(test) > 1 or test[0] != target_length:
        for tr in st:
            diff = tr.stats.npts - target_length
            if abs(diff) >= sample_tolerance * 2.:
                print(f"Trace length mismatch: {tr.stats.npts} vs {target_length} (diff={diff})")
                return None
            if diff > 0:  # trace too long, trim
                tr.data = tr.data[:target_length]
            elif diff < 0:  # trace too short, zero-pad
                tr.data = np.concatenate((tr.data, np.zeros(abs(diff))))
    return st


def add_component_label(channels, component):
    """
    Takes a channel list, adds component label
    """
    new_channels = []
    for chan in channels:
        if chan.isupper:
            chan += component.upper()
        else:
            chan += component.lower()
        new_channels.append(chan)
    return new_channels


def qc_and_preprocess(st, window, sample_tolerance=10):
    """
    Do QC and preprocess waveform data. 
    :param st: Stream including waveform of the given time window
    :type st: SeismonPy Stream object
    :param window: EventWindow object with filter settings
    :type window: EventWindow
    :param sample_tolerance: Tolerance in samples for length/timing differences
    :type sample_tolerance: int
    :return st: Stream including waveform of the given time window or empty stream
    :rtype st: SeismonPy Stream object
    """
    if len(st) == 0:
        return st
    st.sort()
    sampling_rate = st[0].stats.sampling_rate

    for tr in st:
        if tr.stats.sampling_rate != sampling_rate:
            st.remove(tr)
        if np.any(np.isnan(tr.data)):
            st.remove(tr)
    if len(st) == 0:
        print("Removed all traces after sampling rate check.")
        return st
    st.detrend()
    st.taper(window.taper)
    if window.lower_frequency is not None and window.upper_frequency is not None:
        st.filter('bandpass', freqmin=window.lower_frequency, freqmax=window.upper_frequency)

    test = [abs(tr.stats.starttime - st[0].stats.starttime) for tr in st]
    if max(test) > 0:
        if max(test) <= sample_tolerance * 1. / sampling_rate:
            for tr in st:
                tr.stats.starttime = st[0].stats.starttime
        else:
            print(f"Differences in start time larger than {sample_tolerance} samples.")
            return SeismonStream()
    
    window_length = UTCDateTime(window.window_end) - UTCDateTime(window.window_start)
    target_length = int(window_length * sampling_rate)
    st = adjust_trace_length(st, target_length, sample_tolerance)
    if st is None:
        return SeismonStream()

    return st


def get_array_waveforms(window, array_info, client, three_component=True, min_array_stations=3, sample_tolerance=10):
    """
    Load array waveforms for a given time window.
    :param window: EventWindow object
    :type window: EventWindow
    :param array_info: a dictionary including information about seismic arrays
    :type array_info: dict
    :param client: obspy client
    :type client: Client
    :param three_component: If True, load 3-component data, else only Z
    :type three_component: bool
    :param min_array_stations: Minimum number of array stations required for beamforming
    :type min_array_stations: int
    :param sample_tolerance: Tolerance in samples for length/timing differences
    :type sample_tolerance: int
    :return arraystream: Stream including beam waveform of the given time window
    :rtype arraystream: SeismonPy Stream object
    :return array: which array
    """
    station = window.station
    window_start = UTCDateTime(window.window_start)
    window_end = UTCDateTime(window.window_end)

    array = None
    if station in array_info:
        array = station
        array_stations = array_info[array]['Stations']
    else:
        print(f'Station {station} is currently not included in hard-coded array information.')
        return SeismonStream(), None, False
    channels = None
    for t_id, time_period in enumerate(array_info[array]['Time Periods']):
        if time_period[1] == '-':
            time_period[1] = UTCDateTime()
        if (window_start > UTCDateTime(time_period[0]).timestamp and
                window_end < UTCDateTime(time_period[1]).timestamp):
            channels = array_info[array]['Channel Names'][t_id]
            break
    if channels is None:
        print(f'Channels not defined at {window_start}.')
        return SeismonStream(), array, False

    if not three_component:
        channels = add_component_label(channels, 'Z')
    else:
        channels = add_component_label(channels, '*')

    # If more than one channel per station and component, take only first in channel list
    st = SeismonStream()
    basename = []
    for station in array_stations:
        basename.append(station[:2])
    channel_list = ','.join(channels)
    do_single = False
    if len(list(set(basename))) == 1:
        # This is faster than station by station or giving station list:
        try:
            st = client.get_waveforms(basename[0] + '*', channel_list, window_start, window_end)
            for i, station in enumerate(array_stations):
                if three_component and len(st.copy().select(station=station)) != 3:
                    for tr in st.select(station=station)[3:]:
                        st.remove(tr)
                if not three_component and len(st.copy().select(station=station)) != 1:
                    for tr in st.select(station=station)[1:]:
                        st.remove(tr)
                for tr in st:
                    if tr.stats.station not in array_stations:
                        st.remove(tr)
        except (ValueError, PermissionError):
            print("Error during loading. Trying loading channel by channel ...")
            do_single = True
    if len(list(set(basename))) != 1 or do_single:
        st = SeismonStream()
        for i, station in enumerate(array_stations):
            st_tmp = SeismonStream()
            for chan in channels:
                try:
                    st_tmp += client.get_waveforms(station, chan, window_start, window_end)
                except (ValueError, PermissionError):
                    continue
                if three_component and len(st_tmp) == 3:
                    break
                if not three_component and len(st_tmp) == 1:
                    break
            st += st_tmp
    st = qc_and_preprocess(st, window, sample_tolerance)
    if len(st) == 0:
        print(f'Not enough array stations after qc at {window_start}.')
        return st, array, False

    st_z = st.select(component='z')
    nchannels = len(st_z)
    st_z.qc()
    
    st_z.remove_masked()
    if nchannels - len(st_z) > 0:
        print(f'Removed {nchannels-len(st_z)} channels after array qc at {window_start}')
    if len(st_z) < min_array_stations:
        print(f'Not enough array stations after qc at {window_start}.')
        return SeismonStream(), array, False
    if three_component:
        st_n = st.select(component='n')
        st_e = st.select(component='e')
        nchannels = len(st_n) + len(st_e)
        st_n.qc()
        st_n.remove_masked()
        st_e.qc()
        st_e.remove_masked()
        if (nchannels - (len(st_n) + len(st_e))) > 0:
            print(f'Removed {nchannels-(len(st_n)+len(st_e))} horizontal channels after array qc at {window_start}')
        if (len(st_n) < min_array_stations or len(st_e) < min_array_stations):
                print(f'Not enough array stations after qc for horizontal components at {window_start}.')
                return SeismonStream(), array, False
    st = SeismonStream()
    if three_component:
        st = st_z + st_n + st_e
    else:
        st = st_z
    # sort again so that added N and E are ordered and stream can be reproduced
    return st.sort(), array


def get_beam_waveform(window, array_info, client, three_component=True, use_p_for_radial=False, sample_tolerance=10, min_array_stations=3):
    """
    Load three-component beam waveforms (Z,R,T) for a given time window.
    :param window: EventWindow object
    :type window: EventWindow object
    :param array_info: a dictionary including information about seismic arrays
    :type array_info: dict
    :param client: obspy client
    :type client: Client
    :param three_component: If True, create 3-component beams (Z,R,T), else only Z beams
    :type three_component: bool
    :param use_p_for_radial: If True, use P velocity for radial beam, else use S velocity
    :type use_p_for_radial: bool
    :param sample_tolerance: Start and end time difference tolerance in samples (can occur when reading from db)
    :type sample_tolerance: int
    :param min_array_stations: Minimum number of array stations required for beamforming
    :type min_array_stations: int
    :return beam: Stream including beam waveform of the given time window
    :rtype beam: SeismonPy Stream object
    """
    st, array = get_array_waveforms(window, array_info, client, three_component, min_array_stations, sample_tolerance)
    if len(st) == 0:
        return SeismonStream()

    window_start = UTCDateTime(window.window_start)
    baz_p = window.baz
    baz_s = window.baz
    p_vel = window.p_beam_vel
    s_vel = window.s_beam_vel

    inv = client.get_array_inventory(array, window_start)
    sampling_rate = st[0].stats.sampling_rate

    beam = SeismonStream()
    time_delays = inv.beam_time_delays(azimuth_deg=baz_p, velocity_km_sec=p_vel)
    st_z = st.select(component='z')
    st_n = st.select(component='n')
    st_e = st.select(component='e')
    try:
        beam += st_z.create_beam(time_delays=time_delays)
    except ValueError:
        print('Something went wrong unexpectedly during beamforming!')
        return SeismonStream()
    beam[0].stats.channel = 'P-beam, Z'
    beam[0].stats.station = array
    if not three_component:
        time_delays = inv.beam_time_delays(azimuth_deg=baz_s, velocity_km_sec=s_vel)
        beam += st_z.create_beam(time_delays=time_delays)
        beam[-1].stats.channel = 'S-beam, Z'
        beam[-1].stats.station = array
        # beams are mostly longer than input
        beam.trim(st_z[0].stats.starttime, st_z[0].stats.endtime)
    else:
        st3c = st_z + st_n + st_e
        st3c.rotate(method="->ZNE", inventory=inv)
        st3c.rotate(method="NE->RT", back_azimuth=baz_p)
        if use_p_for_radial:
            r_vel = p_vel
            r_baz = baz_p
        else:
            r_vel = s_vel
            r_baz = baz_s
        time_delays = inv.beam_time_delays(azimuth_deg=r_baz, velocity_km_sec=r_vel)
        beam += st3c.select(component='r').create_beam(time_delays=time_delays)
        beam[-1].stats.channel = 'S-beam, R'
        beam[-1].stats.station = array
        time_delays = inv.beam_time_delays(azimuth_deg=r_baz, velocity_km_sec=s_vel)
        beam += st3c.select(component='t').create_beam(time_delays=time_delays)
        beam[-1].stats.channel = 'S-beam, T'
        beam[-1].stats.station = array
        beam.trim(st3c[0].stats.starttime, st3c[0].stats.endtime)

    window_length = UTCDateTime(window.window_end) - UTCDateTime(window.window_start)
    target_length = int(window_length * sampling_rate)
    beam = adjust_trace_length(beam, target_length, sample_tolerance)
    if beam is None:
        return SeismonStream()

    if len(beam) == 0:
        print(f'Empty beam at {window_start}')
        return SeismonStream()
    return beam


if __name__ == "__main__":
    """
    Example usage of get_beam_waveform which load array waveforms does qc, pre-processing (taper, bandpass filter, rotation)
    and beamforming. For three-component data, it creates Z, R, T beams. For single-component data, it creates Z beam.
    For three-component data, it uses P velocity for radial beam and S velocity for horizontal beams.
    For single-component data, it uses S velocity for radial beam and horizontal beams.
    It also removes masked traces and traces with different sampling rates.
    It also removes traces with different start times.
    It also removes traces with different end times.
    It also removes traces with different data.
    It also removes traces with different headers.
    """
    from seismonpy.norsardb import Client
    
    # Initialize client (seismonpy database client)
    client = Client()
    
    # Get array information
    array_info = get_array_info()
    
    # Set up EventWindow with all required parameters
    window = EventWindow(
        station='ARCES',                    # Array name
        window_start='2023-01-01T00:00:00', # Start time (UTC)
        window_end='2023-01-01T01:00:00',   # End time (UTC)
        baz=220.0,                           # Back-azimuth in degrees
        p_beam_vel=6.45,                     # P-wave velocity for beamforming (km/s)
        s_beam_vel=3.7,                     # S-wave velocity for beamforming (km/s)
        lower_frequency=2.0,                # Lower frequency for bandpass filter (Hz)
        upper_frequency=10.0,               # Upper frequency for bandpass filter (Hz)
        taper=0.01                          # Taper fraction (default 0.05)
    )
    
    # Get beam waveforms
    beam = get_beam_waveform(
        window=window,
        array_info=array_info,
        client=client,
        three_component=True,      # True for Z,R,T beams; False for Z only
        use_p_for_radial=False,    # Use S velocity for radial beam (default)
        sample_tolerance=10,       # Tolerance for trace length differences
        min_array_stations=3       # Minimum stations required
    )
    
    # Check result
    if len(beam) > 0:
        print(f"Successfully created {len(beam)} beam trace(s):")
        for tr in beam:
            print(f"  - {tr.stats.station}.{tr.stats.channel}: {tr.stats.npts} samples")
        print(beam)
        beam.plot()
    else:
        print("No beam waveforms returned")
