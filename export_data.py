import os.path
import math
from argparse import ArgumentParser
from glob import iglob
import json
import warnings
from pymongo import MongoClient
from seismonpy.norsardb import Client
from seismonpy.io.nordic import read_nordic
from obspy import UTCDateTime, Stream, Trace
from obspy.io.json.default import Default
from obspy.geodetics import gps2dist_azimuth
import numpy as np
import h5py
import ray


ARCES_LAT = 69.53
ARCES_LON = 25.51


def save_events(query, outpath, recreate):
    """
    Get all event start times, send to remote for processing
    """
    
    dbclient = MongoClient('mongodb://localhost:27020')
    collection = dbclient.events.events
    cursor = collection.find(query)
    
    inv = Client().get_array_inventory('ARCES')
    stored_inv = ray.put(inv)

    futures = []
    for event in cursor:

        futures.append(
            save_waveform.remote(
                event,
                outpath,
                stored_inv,
                recreate
            )
        )
    
    results = {'Success': 0}
    for res in ray.get(futures):
        if res not in results:
            results[res] = 0
        else:
            results[res] += 1

    n_succeeded = results['Success']
    n_failed = len(futures) - n_succeeded
    
    print(f'{n_succeeded} files saved, {n_failed} events failed')
    print('Error count:')
    for res, count in results.items():
        print('  ', res, ':', count)



@ray.remote(max_calls=3)
def save_waveform(event_repr, outpath, inventory, recreate=False):
    """
    Save waveform data
    """

    #warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    
    p_vel = 6.45
    s_vel = 3.70
    edge = 10

    event_time = event_repr['datetime']
    startt = None
    event_repr['analyst_pick_time'] = None

    # Check if analyst pick exists
    if 'picks' in event_repr:
        for pick in event_repr['picks']:
            if 'P' in pick['phase_hint']:
                pick_time = UTCDateTime(pick['time'])
                if startt is not None and pick_time < startt:
                    startt = pick_time
                    event_repr['analyst_pick_time'] = pick_time.__str__()


    if startt is None:
        startt = UTCDateTime(event_repr['est_arrivaltime_arces'])
    
    head_start = 60
    length = 240 # 4 min
    startt -= head_start
    endt = startt + length

    baz = event_repr['baz_to_arces']

    # Set file name, check for existing
    outfile = os.path.join(outpath, startt.__str__()) + '.h5'
    outfile = outfile.replace(':', '.')  # for Windows compatibility
    if os.path.exists(outfile) and not recreate:
        print(f'{outfile} exists, skipping')
        return 'Already existing'

    try:
        comp = 'BH*'
        zcomp = '*Z'
        tcomp = '*T'
        rcomp = '*R'
        if startt < UTCDateTime('2014-09-19T00:00:00'):
            comp = 's*'
            zcomp = '*z'

        stream = Client().get_waveforms(
            'AR*', comp, starttime=(startt - edge), endtime=(endt + edge), sampling_rate_tolerance=0.5
        )

        stream = correct_trace_start_times(stream)

        # Check for masked data, NaN values in traces
        # Remove traces with more than 5 s masked
        masked_traces = []
        for tr in stream.traces:
            if isinstance(tr.data, np.ma.masked_array):
                time_filled = tr.stats.delta* np.sum(tr.data.mask)
                if time_filled > 5.0:
                    print(f'{time_filled:.4f} s of trace data masked, dropping trace - {tr.stats.starttime.__str__()}')
                    masked_traces.append(tr)
                else:
                    tr.data = tr.data.filled(0.0)
                    print(f'{time_filled:.4f} s of trace data masked, filling with zeros - {tr.stats.starttime.__str__()}')

            num_nans = np.sum(np.isnan(tr.data))
            if num_nans > 0:
                time_containing_nans = num_nans*tr.stats.delta
                if time_containing_nans > 5.0:
                    print(f'{time_containing_nans:.4f} s of trace has NaNs, dropping trace - {tr.stats.starttime.__str__()}')
                    masked_traces.append(tr)
                else:
                    tr.data = np.nan_to_num(tr.data)
                    print(f'{time_containing_nans:.4f} s of trace has NaNs, filling with zeros - {tr.stats.starttime.__str__()}')


        for tr in masked_traces:
            if tr in stream.traces: # May have been removed already
                stream.remove(tr)
        
        if len(stream) == 0:
            raise RuntimeError('Stream has no remaining traces')

        stream.detrend('demean')
        stream.taper(max_percentage=None, max_length=edge, type='cosine', halfcosine=True)
        stream.filter('highpass', freq=1.5)
        stream.resample(40.0)
        stream.rotate('NE->RT', back_azimuth=baz, inventory=inventory)

        p_time_delays = inventory.beam_time_delays(baz, p_vel)
        p_beam_z = stream.select(channel=zcomp).create_beam(p_time_delays)
        p_beam_z.stats.channel = 'P-beam, Z'

        s_time_delays = inventory.beam_time_delays(baz, s_vel)
        s_beam_t = stream.select(channel=tcomp).create_beam(s_time_delays)
        s_beam_t.stats.channel = 'S-beam, T'
        s_beam_r = stream.select(channel=rcomp).create_beam(s_time_delays)
        s_beam_r.stats.channel = 'S-beam, R'

        p_beam_z.trim(startt, endt)
        s_beam_t.trim(startt, endt)
        s_beam_r.trim(startt, endt)

        tracedata = np.array([p_beam_z.data, s_beam_t.data, s_beam_r.data])

        # Remove unnecessary info
        event_repr.pop('_id')
        event_repr.pop('datetime')
        event_repr.pop('datetime_inserted')

        # Convert any remaining UTCDateTimes to strings
        event_repr['est_arrivaltime_arces'] = event_repr['est_arrivaltime_arces'].__str__()        

        # Add trace stats
        event_repr['trace_stats'] = {
            'starttime': p_beam_z.stats.starttime.__str__(),
            'sampling_rate': p_beam_z.stats.sampling_rate,
            'station': 'ARCES beam',
            'channels': ['P-beam, vertical', 'S-beam, transverse', 'S-beam, radial']
        }

        evinfo = json.dumps(event_repr)

        with h5py.File(outfile, 'w') as fout:
            fout.create_dataset('traces', data=tracedata, dtype=np.float32)
            string_dt = h5py.special_dtype(vlen=str)
            fout.create_dataset('event_info', data=np.array(evinfo, dtype='object'), dtype=string_dt)

        print(f'Saved {outfile}')
        return 'Success'

        # Debug
        #Stream([p_beam_z, s_beam_t, s_beam_r]).plot()

    except Exception as exc:
        print('ERROR: {} - {}'.format(event_time, exc))
        return str(type(exc)) + str(exc)




def save_noise_events(num, outpath):
 
    strt = UTCDateTime('1998-08-01T00:00:00')
    end = UTCDateTime('2020-11-10T00:00:00')

    dbclient = MongoClient('mongodb://localhost:27020')
    collection = dbclient.events.events
    
    inv = Client().get_array_inventory('ARCES')
    stored_inv = ray.put(inv)
     
    futures = []
    while len(futures) < num:
    
        # Pick a time at random
        tt = np.random.randint(
            low=strt.timestamp,
            high=end.timestamp
        )
        tt = UTCDateTime(tt) 
        
        # Check for registered events
        qry = {
            'est_arrival_time_arces' : {
                '$lt': (tt - 300).datetime,
                '$gt': (tt - 120).datetime
            }
        }
        reg_event = collection.find_one(qry)
        if reg_event is not None:
            continue
        
        event = {
            '_id': 'null',
            'datetime': tt.datetime,
            'datetime_inserted': tt.datetime,
            'est_arrivaltime_arces': tt.datetime,
            'baz_to_arces': np.random.uniform(0.0, np.pi*2),
        }
        
        futures.append(save_waveform.remote(event, outpath, stored_inv))

    results = {'Success': 0}
    for res in ray.get(futures):
        if res not in results:
            results[res] = 0
        else:
            results[res] += 1

    n_succeeded = results['Success']
    n_failed = len(futures) - n_succeeded
    
    print(f'{n_succeeded} files saved, {n_failed} events failed')
    print('Error count:')
    for res, count in results.items():
        print('  ', res, ':', count)





def correct_trace_start_times(stream, max_delta=0.15):
    """
    For old data the traces might have tiny offset in start time, which breaks
    beamforming. Adjust this manually.
    Remove traces with diff > max_delta
    """
    sts = [tr.stats.starttime for tr in stream.traces]
    most_common = np.unique(sts)[0]

    for tr in stream.traces:
        this_starttime = tr.stats.starttime
        if this_starttime != most_common:
            if abs(this_starttime - most_common) <= max_delta:
                tr.stats.starttime = most_common
            else:
                print('Removing trace:', tr)
                stream.remove(tr)
    
    return stream


def insert_datetime(d):
    # Search for dates and convert string to datetime object
    import re
    newdict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            newdict[k] = insert_datetime(v)
        else:
            if isinstance(v, str) and re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', v):
                v = UTCDateTime(v).datetime
            newdict[k] = v
    return newdict



def get_dprk_waveforms():
    # Use USRK, ARCES, MJAR, KSRS
    
    client = Client()

    events = {
        'NK1': {
            'time': UTCDateTime('2006-10-09T01:35:27'),
            'mag': 4.2
        },
        'NK2': {
            'time': UTCDateTime('2009-05-25T00:54:43'),
            'mag': 4.7
        },
        'NK3': {
            'time': UTCDateTime('2013-02-12T02:57:51'),
            'mag': 5.0
        },
        'NK4': {
            'time': UTCDateTime('2016-01-06T01:30:01'),
            'mag': 4.9
        },
        'NK5': {
            'time': UTCDateTime('2016-09-09T00:30:01'),
            'mag': 5.2
        },
        'NK6': {
            'time': UTCDateTime('2017-09-03T03:30:02'),
            'mag': 6.1
        }
    }
        #UTCDateTime('2006-10-09T01:35:27'),  #NK1
        #UTCDateTime('2009-05-25T00:54:43'),  #NK2
        #UTCDateTime('2013-02-12T02:57:51'),  #NK3
        #UTCDateTime('2016-01-06T01:30:01'),  #NK4
        #UTCDateTime('2016-09-09T00:30:01'),  #NK5
        #UTCDateTime('2017-09-03T03:30:02'),  #NK6
    
    time_to_station = {
        'USRK': 57,
        'MJAR': 125,#60,
        'KSRS': 62,
        'ARCES': 581
    }
    station_name_pattern = {
        'USRK': 'USA*,USB*',
        'MJAR': 'MJA*,MJB*',
        'KSRS': 'KS0*,KS1*,KS2*,KS3*',
        'ARCES': 'ARA*,ARB*,ARC*,ARD*'
    }
    station_channel_pattern = {
        'USRK': 'SHZ,SZ',
        'MJAR': '*',
        'KSRS': 'S*,B*',
        'ARCES': 'BH*'
    }
    station_coords = {
        'USRK': (44.1998, 131.9888),
        'MJAR': (36.5247, 138.2472),
        'KSRS': (37.4421, 127.8844),
        'ARCES': (ARCES_LAT, ARCES_LON)
    }
    station_appvel_p = {
        'USRK': 7.4,
        'MJAR': 6.2,
        'KSRS': 8.4,
        'ARCES': 12.7,
    }
    testsite_coords = (41.299, 129.075)

    zcomp = '*Z'
    tcomp = '*T'
    rcomp = '*R'
    edge = 10
    head_start = 60
    length = 240 # 4 min
    
    for station in station_name_pattern:



        _, baz, _ = gps2dist_azimuth(
            station_coords[station][0],
            station_coords[station][1],
            testsite_coords[0],
            testsite_coords[1]
        )
        
        try:
            inventory = client.get_array_inventory(station)
        except ValueError as exc:
            print(station, exc)
            inventory = None

        
        for eventname in events:

            tt = events[eventname]['time']

            if station == 'USRK' and tt < UTCDateTime(2009, 1, 1):
                continue

            startt = tt + time_to_station[station] - head_start
            endt = tt + time_to_station[station] + length

            comp = station_channel_pattern[station]

            # Channel selection for ARCES
            if station == 'ARCES' and tt < UTCDateTime('2014-09-19T00:00:00'):
                comp = 's*'
                zcomp = '*z'

            # Reload inventory for MJAR
            if inventory is None or station == 'MJAR':
                inventory = client.get_array_inventory(station, time=tt)

            stream = Client().get_waveforms(
                station_name_pattern[station],
                comp,
                starttime=(startt - edge),
                endtime=(endt + edge),
                sampling_rate_tolerance=0.5
            )

            print(station, tt)
            print('Before processing:')
            print(stream.__str__(extended=True))

            stream.remove_masked()
            stream.detrend('demean')
            stream.taper(max_percentage=None, max_length=edge, type='cosine', halfcosine=True)
            stream.filter('highpass', freq=1.5)
            stream.resample(40.0)
            stream.rotate('NE->RT', back_azimuth=baz, inventory=inventory)

            print('After processing:')
            print(stream.__str__(extended=True))

            # Vertical P beam
            p_time_delays = inventory.beam_time_delays(baz, station_appvel_p[station])
            p_beam_z = stream.select(channel=zcomp).create_beam(p_time_delays)
            p_beam_z.stats.channel = 'P-beam, Z'

            # Radial, transverse "S" beam - ARCES only
            if station == 'ARCES':
                s_time_delays = inventory.beam_time_delays(baz, 18.6)
                s_beam_t = stream.select(channel=tcomp).create_beam(s_time_delays)
                s_beam_t.stats.channel = 'S-beam, T'
                s_beam_r = stream.select(channel=rcomp).create_beam(s_time_delays)
                s_beam_r.stats.channel = 'S-beam, R'

            # Radial, transverse single components
            elif station == 'KSRS' or station == 'MJAR':
                s_beam_t = stream.select(channel=tcomp).traces[0]
                s_beam_t.stats.channel = 'S-beam, T'
                s_beam_r = stream.select(channel=rcomp).traces[0]
                s_beam_r.stats.channel = 'S-beam, R'

            # USRS has only Z channels
            else:
                stats = dict(stream.traces[0].stats.__dict__)
                stats['channel'] = 'S-beam, T'
                s_beam_t = Trace(
                    data=np.zeros(shape=stats['npts']),
                    header=stats
                )
                stats['channel'] = 'S-beam, R'
                s_beam_r = Trace(
                    data=np.zeros(shape=stats['npts']),
                    header=stats
                )

            Stream([p_beam_z, s_beam_t, s_beam_r]).plot()

            p_beam_z.trim(startt, endt)
            s_beam_t.trim(startt, endt)
            s_beam_r.trim(startt, endt)

            tracedata = np.array([p_beam_z.data, s_beam_t.data, s_beam_r.data])


            # Fill eventdata 
            event_repr = {
                'event_type': 'nuclear explosion',
                'event_type_certainty': 'known',
                'origins': [
                    {
                        'time': tt.__str__(),
                        'longitude': 41.299,
                        'latitude': 129.075,
                        'depth': 0.0,
                        'region': 'Punggye-ri test-site'
                    }
                ],
                'magnitudes': [
                    {
                    'mag': events[eventname]['mag'],
                    'magnitude_type': 'MB'

                    }
                ]

            }
            event_repr['est_arrivaltime'] = (tt + time_to_station[station]).__str__()

            # Add trace stats
            event_repr['trace_stats'] = {
                'starttime': p_beam_z.stats.starttime.__str__(),
                'sampling_rate': p_beam_z.stats.sampling_rate,
                'station': f'{station} beam',
                'channels': ['P-beam, vertical', 'S-beam, transverse', 'S-beam, radial']
            }

            evinfo = json.dumps(event_repr)

            outfile = eventname + '_' + tt.__str__() + '_' + station + '.h5'
            outfile = outfile.replace(':', '.')

            with h5py.File(outfile, 'w') as fout:
                fout.create_dataset('traces', data=tracedata, dtype=np.float32)
                string_dt = h5py.special_dtype(vlen=str)
                fout.create_dataset('event_info', data=np.array(evinfo, dtype='object'), dtype=string_dt)

            print(f'Saved {outfile}')


            #stream = correct_trace_start_times(stream)
            if len(stream) == 0:
                print(station, tt, ': Empty stream')
                continue

            #stream.plot()






if __name__ == '__main__':


    parser = ArgumentParser('Export events to h5 files')
    parser.add_argument('-pl', '--plot', action='store_true')
    parser.add_argument('-s', '--save', type=str, help='Output directory', default=None)
    parser.add_argument('-q', '--query', type=str, help='JSON string', default='{}')
    parser.add_argument('-sn', '--save_noise', action='store_true')
    parser.add_argument('-nk', '--north_korea', action='store_true')
    args = parser.parse_args()

    args.query = json.loads(args.query)
    args.query = insert_datetime(args.query)

    if args.plot:
        #debug_data_plot(args.query)
        debug_beam_plot(args.query)

    if args.save is not None:
        ray.init(num_cpus=8)
        assert os.path.isdir(args.save)
        save_events(args.query, args.save, False)

    if args.save_noise:
        ray.init(num_cpus=8)
        assert os.path.isdir('/nobackup/steffen/data_tord_nov2020/noise')
        save_noise_events(120000, '/nobackup/steffen/data_tord_nov2020/noise')

    if args.north_korea:
        get_dprk_waveforms()