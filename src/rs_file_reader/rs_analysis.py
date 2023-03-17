'''
    File name: fileReader.py
    Author: Simon Waid
    Date created: 2021
    Python Version: 3.8, 3.9
'''

import xml.etree.ElementTree as ET

import os
import logging
import numpy as np
import sys
import time
import math
from numba import njit
#import vaex
import hashlib
import numpy as np
import matplotlib.pylab as plt
import scipy.integrate as integrate
import json
import pickle

#from tool_box import ProcessingHelper
import pandas as pd

class Osci_XML_Tags():
    '''This is an empty class for making the oscilloscope configuration stored in the xml accessible'''
    def __init__(self):
        pass
    
class MyCache:
    '''
    Implements caching for :py:class:`Det_File`
    '''
    ENABLE_IN_RAM = False
    def __init__(self, cache_dir, instance_id):
        '''
        :param class_hash: Data that make the instance unique.
        '''
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.mkdir(self.cache_dir)
        self.instance_id=instance_id.encode('utf-8')
        self.data={}

    def cache(self, func):
        '''
        Decorator for cachings.
        ''' 
        def cache_func(*args, **kwargs):
            hash = hashlib.sha1()
            # Include data about then instance 
            hash.update(self.instance_id)
            # Skip self.
            # TODO: Improve this.
            args = list(args)
            for i in range(len(args)):
                if type(args[i]) == pd.DataFrame:
                    args[i] = args[i].to_json()
            cache_key = json.dumps([func.__name__, args, sorted(kwargs)], separators = (',', ':'))
            hash.update(cache_key.encode('utf-8'))
            my_hash = hash.hexdigest()
            # Access the in-RAM cache first
            if hash in self.data.keys():
                
                print(f'In-RAM-Cache hit {func.__name__}, hash: {my_hash}')
                return self.data[my_hash]
            else:
                hashFileName = os.path.join(self.cache_dir, my_hash + '.pickle')
                # Try to access the on Disk cache.
                if os.path.exists(hashFileName):
                    try:
                        with open(hashFileName, 'rb') as f:
                            result=pickle.load(f)
                        print(f'On-Disk-Cache hit {func.__name__}, hash: {my_hash}')
                    except:
                        print(f'Failed to load from disk-cache. Consider implementing an automated handling of this situation.,')
                        raise
                else:
                    print(f'Cache miss {func.__name__}, hash: {my_hash}')
                    # Execute the function an store the data in the on-disk cache
                    t = time.time()
                    result = func(*args, **kwargs)
                    
                    print('Computation took {:.2}s'.format(time.time() - t))
                    with open(hashFileName, 'wb') as f:
                        pickle.dump(result, f)
                    print(f'Stored in cache {my_hash}')
                # Store in the In-RAM cache.
                if self.ENABLE_IN_RAM:
                    self.data[hash] = result
                
                return result
            
        return cache_func
                
class RS_Analysis():
    '''
    Extends RS_File with analysis functions for detectors.
    '''
    BUFFER = 5e-8
    def __init__(self, rs_file, cache_dir = None):
        '''
        :param rs_file: Instance of :py:class:`RS_file`
        :param cache_dir: Optional. If given, some computationally expensive functions will use a cache. This requires joblib. 
        '''
        self.rs_file=rs_file
        # Initialize the cache
        # Change this rev. number to invalidate the cache
        rev = 1
        self.cache_fast_stat={}
        self.cache_statistics={}
        if not cache_dir is None:
            instance_id = str(self.rs_file.meta)+str(rev)
            self.memory = MyCache(cache_dir, instance_id)
            # Overwrite the functions for which we would like to use the cache.
            self.__getTot = self.getTot
            self.getTot = self.memory.cache(self.__getTot)
            
            self.__getStatistics = self.getStatistics
            self.getStatistics=self.memory.cache(self.__getStatistics)
            
            self.__fastStat = self.fastStat
            self.fastStat=self.memory.cache(self.__fastStat)
        
        else:
            self.memory = None

    
    def fastStat(self, sources=None, offset_meth='avg'):
        '''
        Fast calculation of statistics over the full sample.
        :TODO: Implement multiprocessing.
        
        :param sources:
        :param offset_meth: Optional, defaults to "avg". Available options are "avg", "median", "oscilloscope". 
        
        :returns: dict containing statistics.
        '''
        
        sources=self.rs_file._sanitize_sources(sources)
        
        # We cache the result of each function call and perform the statistics computation only if needed.
        key=f'sources: {sources}, offset_meth: {offset_meth}'
        if key in self.cache_fast_stat:
            return self.cache_fast_stat[key]
        
        result={}
        # Calculation loop.
        # Should be parallelized
        std_res=[]
        for source in sources:
            
            if not source in self.rs_file.data_raw_RS.keys():
                raise(RuntimeError(f'Invalid source {source}'))
            raw_data=self.rs_file.data_raw_RS[source]
                
            # Depending on the data format use a different method for calculating the statistics.
            # If we have raw adc counts we use  _stat_raw.
            # If we have floats we use  _stat.
            if offset_meth == 'avg':
                offset_signal=np.average(raw_data)
            elif offset_meth == 'median':
                offset_signal=np.median(raw_data)
            elif offset_meth == 'oscilloscope':
                offset_signal=0
            else:
                raise(RuntimeError('Invalid offset method. Available options: "avg" and "median"'))
            
            if self.rs_file.meta['apply_offset']:
                scaling = self.rs_file.meta['conversion_factor'][source]
                if offset_meth == 'oscilloscope': 
                    offset =self.rs_file.meta['offset'][source]
                    offset_ret=0.
                else:
                    offset = 0.
                    offset_ret=offset_signal*scaling+self.rs_file.meta['offset'][source]
                # Calculate the standard deviations
                data_ch = _stat_raw(raw_data, offset, offset_signal, scaling)
                std_res.append(data_ch)
                
                #  p.add(_one_side_std_raw, [raw_data, offset, scaling])
            else:
                std_res.append(_stat(raw_data, offset_signal))
                offset_ret=offset_signal
                #    p.add(_one_side_std, [raw_data])
        
        # Evaluation loop
        for i, source in enumerate(sources):
            
            stdA, stdB, my_min, my_max= std_res[i]
            # Out standard deviation for the threshold is the minimum. 
            std=min([stdA, stdB])
            result[source]={}
            result[source]['std'] = std
            result[source]['stdA'] = stdA
            result[source]['stdB'] = stdB
            result[source]['min'] = my_min
            result[source]['max'] = my_max
            result[source]['offset'] = offset_ret

        self.cache_fast_stat[key] = result
                
        return result
 
    def getStatistics(self, acquisition=None, start = None, stop = None):#, source=None):
        '''
        Returns some statistics on the data. This is a slow but reliable implementation. Consider using :py:meth:`.fastStat`
        
        :param start: See :py:meth:`.getAsDf` 
        :param stop: See :py:meth:`.getAsDf`
        '''
        #:param source: Oscilloscope Source Name
        
        #raise(RuntimeError('This function is buggy, please fix is p'))
        
        # Get filtered data
        data = self.rs_file.getRaw(acquisition=acquisition, start = start, stop = stop, source=None)
        
        result = {}
        
        # We operate on one channel at a time to keep the memory footprint low.
        for i, ch in enumerate(self.rs_file.meta['source_names']):

            result[ch] = {} 
            
            # Convert the data to floating point.
            # We generate a dict that looks like the raw_data dict so we can use the same functions.
            data_ch={ch:data[ch]} 
            data_ch = self.rs_file.rawToDtypeOut(data_ch)[ch]

            # Check over the whole sample whether peaks are on the negative or the positive 
            # side. 
            my_min = np.amin(data_ch)#.astype(np.int64)
            my_max = np.amax(data_ch)#.astype(np.int64)
            
            result[ch]['min'] = my_min
            result[ch]['max'] = my_max
                        
            # If only fast tasks were requested we're done!
            # Otherwise it's time to calculate the standard deviation.
            # Fast am memory efficient implementation of std calculation.
            # Apply a filtering only using values on the other side of the peaks to calculate the std
            if np.abs(my_min) > np.abs(my_max):
                mask= data_ch<=0
            else:
                mask= data_ch>=0
            
            masked=data_ch[mask]
            del mask
            # Time and memory efficient rms over the masked entries 
            masked=np.square(masked)
            std = np.sqrt(np.mean(masked))
            del masked
            
            result[ch]['std'] = std
            # Convert output 
            
        return result
    
    def get_2d_histo(self, source, start = None, stop = None, no_bins_x=8000, max_bins_y=1024, cutoff_y=1E-5):
        '''
        Returns a 2D histogram of the data. This is useful for plotting acquisitions with a large number of samples.
        
        :param source: Oscilloscope Source Name. 
        :param start: Index of starting sample
        :param stop: Index of final sample
        :param no_bins_x: Number of bins in x direction. Optional, defaults to 8192 (fits an 8k screen, the largest type of screen currently on the market).
        :param max_bins_y: Maximum number of bins in y direction. Optional, defaults to 1024. The resolution of the ADC will be inferred from the data. The number of y bins will be adapted to the inferred resolution.   
        :param cutoff_y:
        
        :returns: histogram, extent where histogram is a 2d numpy array and extent contains the limits in physical units for use with matplotlib imshow.
                
        '''
        
        data = self.rs_file.getRaw(start = start, stop = stop, source=source)
        #data=data[source]
        # We need to parameterize our 2D output array first.
        # We start with the number of y bins. 
        # We compute the unique values first.
        unique=np.unique(data)
        # And find find the minimum distance between unique values.
        dist_y=np.min(unique[1:]-unique[:-1]).astype(np.float64)
        # Assuming the ADC is linear we now know it's resolution.
        # Now let's get the minimum and maximum values so we known the swing of the data.
        max_val=np.max(data).astype(np.float64)
        min_val=np.min(data).astype(np.float64)
        
        print(min_val, max_val)

        # We now can compute the number of needed bins.
        span=max_val-min_val
        no_bins_y=int(span/dist_y)
        # We might have more bins than the ADC has levels. In that case restrict to the number of ADC levels
        if  no_bins_y > self.rs_file.meta['quantisation_levels']:
            no_bins_y = self.rs_file.meta['quantisation_levels']
        
        # If we need to restrict ourselves, we first attempt to remove outliers.
        if no_bins_y > max_bins_y:
            # To find outliers we create a 1D histogram 
            histo_1d_occ, histo_1d_val=np.histogram(data, bins=no_bins_y)
            # We find the maximum occurrence 
            histo_1d_val=histo_1d_val[:-1]
            max_occ=np.max(histo_1d_occ)
            # Compute the threshold and eliminate everything below the threshold
            threshold=max_occ*cutoff_y
            
            above_th_bool=histo_1d_occ > threshold
            #where_above_th=np.where(above_th_bool)
            above_th=histo_1d_val[above_th_bool]
            # Update the limits of the y part of interest
            max_val=np.max(above_th).astype(np.float64)
            min_val=np.min(above_th).astype(np.float64)
            span=max_val-min_val
            no_bins_y=int(span/dist_y)
            print(threshold, min_val, max_val)
            
        if no_bins_y > max_bins_y:
            no_bins_y = max_bins_y
            
        
        # Update the distance between samples
        dist_y=span/no_bins_y
        
        # In x direction the number of bins is simply limited by the number of samples:
        if len(data) < no_bins_x:
            no_bins_x=len(data) 
        
        #if 'X-Time' in self.rs_file.meta['channel_type']:
            #raise(RuntimeError('Not implemented for explicit time scales'))
        #    time = self.rs_file.getRaw(start = start, stop = stop, source='X-Time')
        #    histogram=np.histogram2d(data, time, bins=[no_bins_x, no_bins_y])
        #else:
        histogram=_histo_implicit_t(data, no_bins_x, no_bins_y, min_val, dist_y)
        # Now create the histogram
        
        if self.rs_file.meta['apply_offset']:
            y_min=min_val*self.rs_file.meta['conversion_factor'][source]+self.rs_file.meta['offset'][source]
            y_max=max_val*self.rs_file.meta['conversion_factor'][source]+self.rs_file.meta['offset'][source]
        else:
            y_min=min_val
            y_max=max_val
    
        #x_axis=np.linspace(self.meta['xStart'], self.meta['xStop'], no_bins_x)
        #y_axis=np.linspace(y_min, y_max, no_bins_y)
        
        if not start is None or not stop is None:
            raise(RuntimeError('Not implemented!'))
             
        extent=[self.rs_file.meta['xStart'], self.rs_file.meta['xStop'], y_min, y_max]
            
        return histogram, extent # x_axis, y_axis
        
    def plotEvents(self, events, dir_out, sources= None, no_events = 10, buffer = BUFFER):
        '''
        :param events: Output form getTot
        :param sources: The oscilloscope Source that should be processed.
        :param dir_out: Directory for plotting.
        :param no_events: Maximum number of events that should be plotted.
        :param buffer: How much data should be plotted before and after the event. Defaults to ns.
        '''
        # If the directory does not exist create it
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        # 
        if sources is None:
            sources= self.rs_file.meta['source_names']
        # If we didn't get a list, assume it's a single source and create a list 
        elif not type(sources) == type([]):
            sources =[sources]
            
        for i, source in enumerate(sources):
            
            ev=events[source]
            # If no events were detected silently skip plotting
            if ev is None:
                continue
            # make local variable for all relevant data.
            start = ev['Start'].values
            length = ev['Duration'].values
            t_sample = self.rs_file.meta['t_sample']
            source_names = source
            buffer_raw = int(buffer / t_sample)

            plot_color = ['blue', 'green', 'orange', 'black']
    
            # Limit the plotting to the first events as specified by no_events
            if len(start) > no_events:
                start = start[:no_events]
                length = length[:no_events]
                
            peak_count = 1
            for s, l in zip(start, length):
                start = s - buffer_raw
                stop = s + l + buffer_raw
                if start < 0:
                    start = 0
                if stop > (self.rs_file.total_no_samples):
                    stop = int(self.rs_file.total_no_samples)
                try:
                    data = self.rs_file.getAsDf(start=start, stop=stop, time=True)
                except:
                    print('Breakpoint!')
                    raise
                time = data['Time']
                # change axis to begin at position of the first sample exceeding the threshold
                time = time - s*t_sample
                voltage = data[source_names]
    
                plt.close()
                plt.clf()
                plt.plot(time, voltage, '-', color = plot_color[i])
    
                plt.grid()
                #plt.legend(title='Parameter where:')
                plt.title(f'{str(source)} - Peak_no: {str(peak_count)}')
                outfile = os.path.join(dir_out, ('peak' + str(peak_count)+'.png'))
                plt.savefig(outfile, dpi = 300)
                peak_count += 1
                plt.close()

    def get_threshold(self, sources, std_factor = 6):
        '''
        Returns the threshold for each channel based on a given dict including max, min and std values as 
        we get it for each channel from _getStatistics
        :param sources: the sources for which the threshold should be computed.
        
        :TODO: Also Handle non-AC-couped signals. 
        '''
        result = {}
        
        stats=self.fastStat(sources)
            
        for source, values in stats.items():
            threshold = 0
            if abs(values['min']) > abs(values['max']):
                threshold = -std_factor * values['std']
            else:
                threshold = std_factor * values['std']
            result[source] = threshold

        return result

    def getTot(self, source, threshold, cmp = None, signal_samples = None, tmin_gap = BUFFER):
        '''

        Get the time over threshold for signals. Analyses the peak signals over the given
        threshold and determines where peaks occured and how long they lasted. Returns a dataframe 
        containing the start indices and the length (in samples) reffering to the raw data. It includes
        a fine tuning process, changing the minimal required signal length time according to the found
        data. 

        :param source: The source name on which the operation should be carried out.
        :param threshold: The threshold for detecting an event. Unit: Volts.
        :param cmp: Either '>' or '<'. Determines if values above or below the threshold should be considered
        :param gap_time: Search limit for the minimal required time between two signals. Unit: seconds
        :param signal_time: Initial search limit for the minimal required time to be considered as signal. Unit: seconds
        
        :returns: Pandas DataFrame. The Columns are Start and Duration. They contain the beginning and end of peaks in sample indices.  
        '''
        
        if cmp is None:
            if threshold < 0:
                cmp = '<'
            else:
                cmp = '>'

        t_sample = self.rs_file.meta['t_sample']
        
        # Set the minimal signal time to at least three samples, if not given specifically
        if signal_samples == None:
            tmin_signal = 2 * t_sample
        else:
            tmin_signal = signal_samples * t_sample

        data = self.rs_file.data_raw_RS[source]
        
        if np.isnan(threshold):
            raise(RuntimeError('Invalid threshold. NaN is now allowed'))
        
        # Convert the threshold to raw data units. 
        if self.rs_file.meta['apply_offset']:
            threshold_raw = threshold - self.rs_file.meta['offset'][source]
            threshold_raw = threshold_raw / self.rs_file.meta['conversion_factor'][source]  
            threshold_raw = np.asarray(threshold_raw, dtype=data.dtype)
        else:
            threshold_raw = threshold

        if tmin_signal < t_sample:
            raise(RuntimeError('Initial signal length is smaller than the sample size!!!'))

        if cmp == '>':
            if threshold < 0:
                logging.warn(f'threshold is below 0 but cmp is >. This is an unexpected combination.')
            det = data >= threshold_raw
        elif cmp == '<':
            if threshold > 0:
                logging.warn(f'threshold is below 0 but cmp is >. This is an unexpected combination.')
            det = data < threshold_raw
        else:
            raise(RuntimeError('Invalid value for cmp'))
            
        # Free memory
        del data 

        det = det.astype(np.int8)
        # Theory of operation: 
        # Lets assume we have the following in a:
        # a = [0 0 0 1 1 1 1 0 0]
        # We shift by 1 to the left:
        # b = a[1:]
        # b = [0 0 1 1 1 1 0 0]
        # Now let's subtract:
        # a[:-1] - b = [ 0 0 -1 0 0 0 1 0]
        
        det2 = det[:-1] - det[1:]
        # Free memory
        del det

        # Get indices of starting and ending points for each peak
        starts_in = np.where(det2 == -1)
        ends_in = np.where(det2 == 1)
        starts = starts_in[0]
        ends = ends_in[0]
        
        # Ensure the start and ends to be in sync
        if len(starts) < 1:
            return None
        if starts[0] > ends[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:len(ends)]

        # Free memory
        del det2

        # Find the corresponding peaks according to some required settings 
        #starts_out, ends_out = _peak_filter(starts, ends, tmin_signal_raw, tmin_gap_raw)
        starts_out=starts
        ends_out=ends

        # Convert to numpy
        starts_out = np.array(starts_out)
        ends_out = np.array(ends_out)
        
        # Now convert starts and ends to starts and length of signals
        result = pd.DataFrame(starts_out, columns=['Start'])
        try:
            result['Duration'] = ends_out - starts_out
        except:
            print('Len start: {}, Len end: {}'.format(len(starts_out), len(ends_out)))
            raise
        result['Duration_SI'] = result['Duration']*self.rs_file.meta['t_sample']
        if np.any(result['Duration'] <= 0):
            raise(RuntimeError("Negative durations don't exist. This is a bug."))

        return result

    def analyse_peaks(self, events, source, cmp, buffer = BUFFER):
        '''
        Analyze single peaks. Takes an events dataframe as returned by the function getTot_fine()
        and returns a dataframe containing the maximal peak heigthh and the peak areas of each event. Has 
        to be launched for each file and channel.
        The method covers different signs of the voltage, such that returned values are always positive 
        and refer to the absolute values of height and area.
    
        :param events: Starting indices and length in samples of the found events from getTot_fine().
        :param source: The Souce  number on which the operation should be carried out.
        :param cmp: < means too look for maxima in the negative direction, > otherwhise.
        :param buffer: Event locations come at time where threshold is exceeded and underpassed again. 
        The actual peak takes longer than that, therefore we add a buffer on both ends. 
        '''  
        ch = self._chNameToNum(source)
        t_sample = self.rs_file.rs_file.meta['t_sample']
        buffer_raw = int(buffer / t_sample)

        results = pd.DataFrame(columns = np.array(['max_height', 'area']))
        max_height = []
        area = []

        if len(events) == 0:
            return None

        start = events['Start'].values
        length = events['Duration'].values
        
        dataList = []
        for s, l in zip(start, length):
            start = s - buffer_raw
            stop = s + l + buffer_raw
            if start < 0:
                start = 0
            if stop > (self.rs_file.meta['length'] - buffer_raw):
                stop = int(self.rs_file.meta['length'] - 1)
            length = stop - start
            timestamps = np.linspace(start, stop, length, dtype = np.float64)
            data = self._getRaw(start, stop, sourceNo = ch)
            data = self._rawToDtypeOut(data, sourceNo = ch)
            
            dataList.append([timestamps, data])

        for data in dataList:
            mytime = data[0]
            voltage = data[1]

            # Check for the direction of the signal voltage and adapt accordingly 
            # The integration method can of course be changed. The simple trapezoidal method is used
            # currently. However, one can change this to simpson() or even higher order integrations.
            if cmp == '<':
                max_height.append(-min(voltage))
                area.append((-integrate.trapz(y = voltage, x = mytime)) * t_sample)
            else :
                max_height.append(max(voltage))
                area.append((integrate.trapz(y = voltage, x = mytime)) * t_sample)
        
        results['max_height'] = np.array(max_height)
        results['area'] = np.array(area)

        return results

    def getTotAuto(self, sources=None, offset_meth='avg', std_threshold=6):
        '''
        Convenience function to get the tot with reasonable assumptions. Calls :py:meth:`.getTot_fine`
        
        :param sources: List of sources to be processed. 
        :param offset_meth: Optional, defaults to "avg". Available options are "avg", "median". 
        :param std_threshold:
        
        :returns: results, meta. results is a dict containing the results. the source name is the key. meta is a dict containing meta data such as the standard deviations. 
        '''
        
        # TODO: Implement offset detection and elimination  
        
        if sources is None:
            sources= self.rs_file.meta['source_names']
        elif type(sources) == list:
            pass
        else:
            raise(RuntimeError('sources must be None or a list'))
        
        # Get the statistics.
        t=time.time()
        
        #stats = self.getStatistics(source=sources)
        stats = self.fastStat(sources, offset_meth=offset_meth)
        print('Statistics took: ', time.time()-t)
        
        # Calculate the time over threshold
        meta=stats
        for source in sources:
            # Get the result from the previous (parallel) calculation
            std=stats[source]['std']
            stdA=stats[source]['stdA']
            stdB=stats[source]['stdB']
            my_min=stats[source]['min']
            my_max=stats[source]['max']
            # Get the threshold.
            threshold= std*std_threshold
            # Adjust the sign of threshold.
            # Using the min and max value to determine the sign seems best.
            # An alternative would be to compare the two standard deviations.
            # However, the latter method can be inaccurate due to numeric issues.
            # We start by comparing min and max.  If there is sufficient difference, we use min and max.
            # If not, we use the stds
            ma= max([abs(my_min), abs(my_max)])
            mi= min([abs(my_min), abs(my_max)])
            if abs(ma/mi-1) <1E-3:
                # Difference between min and max too small 
                if stdA > stdB:
                    threshold*=-1
            elif abs(my_min) > abs(my_max):
                
                threshold*=-1
            print(f'Source: {source} std {std}, min {my_min}, max {my_max} ==> threshold: {threshold}')
            meta[source]['threshold']=threshold
            
        # Calculate the time over threshold 
        results={}

        for source in sources:
            
            results[source] = self.getTot(source, threshold)
            
        return results, meta 

@njit
def _histo_implicit_t(data, no_bins_x, no_bins_y, min_y, dist_y):
    '''
    Compute a histogram of the given data. It assumes there are no time stamps available in the data. If time stamps are available, the numpy histogram2d function can be used.
    
    '''

    # Initialize the histogram. For now  we have zero counts every
    output=np.zeros((no_bins_x, no_bins_y))
    
    # Determine the distance between bins in the x direction.  
    length=len(data)
    dist_x=length/no_bins_x

    #Multiplications are cheaper than divisions, so divide once. Later on use the inverse 
    dist_y_inv=1/dist_y
    
    # Initialize variables for the loop. 
    x_index=0
    x_index_limit=dist_x
    for index, d in enumerate(data):
        # Determine the location of our sample in x direction
        if index > x_index_limit:
            x_index+=1
            x_index_limit= (x_index+1)*dist_x
        # Determine the location of our sample in x direction
            
        y_index=int(float(d-min_y)*dist_y_inv)
        # Avoid out of range indexing
        if y_index  < 0  or y_index > no_bins_y:
            continue
        # Update the histogram
        output[x_index, y_index] +=1
    
    return output
    
@njit
def _stat_raw(data, offset, offset_signal, scaling):
    '''
    Compute statistics from raw data.

    '''
    
    sideA_sum=float(0)
    sideB_sum=float(0)
    sideA_len=float(0)
    sideB_len=float(0)
    
    l=len(data)
    
    my_min=float(data[0]-offset_signal)*scaling+offset
    my_max=float(data[0]-offset_signal)*scaling+offset
    
    for d in data:
        #df=float(d)
        df=float(d-offset_signal)*scaling+offset
        if df <= 0:
            sideA_sum +=df**2
            sideA_len +=1
        if df >= 0:
            sideB_sum +=df**2
            sideB_len +=1
        if df < my_min:
            my_min=df
        if df > my_max:
            my_max=df
             
    if sideA_len == 0:
        stdA = np.nan
    else:     
        stdA = math.sqrt(sideA_sum/sideA_len)
    if sideB_len == 0:
        stdB = np.nan
    else:
        stdB = math.sqrt(sideB_sum/sideB_len)
    
        
    return stdA, stdB, my_min, my_max
    
@njit
def _stat(data, offset_signal):
    '''
    Compute statistics from floating point numbers     
    
    '''
    
    sideA_sum=float(0)
    sideB_sum=float(0)
    sideA_len=float(0)
    sideB_len=float(0)
    
    my_max=data[0]
    my_min=data[0]
    
    for d in data:
        d-= offset_signal
        if d <= 0:
            sideA_sum +=d**2
            sideA_len +=1
        if d >= 0:
            sideB_sum +=d**2
            sideB_len +=1
        if d < my_min:
            my_min=d
        if d > my_max:
            my_max=d
    
    # Calculate std. Avoid division by 0 
    if sideA_len == 0:
        stdA = np.nan
    else:
        stdA = math.sqrt(sideA_sum/sideA_len)
    if sideB_len == 0:
        stdB = np.nan
    else:
        stdB = math.sqrt(sideB_sum/sideB_len)
    
    return stdA, stdB, my_min, my_max
    
@njit
def _peak_filter(starts, ends, lenght_min, tmin_gap_raw):
    '''
    Filters the detected peaks. For use by :py:meth:RS_Det_File.getTot_fine.
        
    :param starts: list of indices with a potential start of a peak
    :param ends: list of indices with a potential end of a peak. Must match starts
    :param lenght_min: minimum number of samples for a peak to valid
    :param tmin_gap_raw: Minimum gap beteen peeks.
        
    :returns starts_out, ends_out: Filtered lists like starts and ends
    '''
    starts_out = []
    ends_out = []
    
    # Init for the loop.
    max_i = len(starts)
    i = 0
    while(True):
        length = ends[i] - starts[i]
        # First condition: The distance between the start and end must be sufficient. 
        if length >= lenght_min:
            start = starts[i]
            starts_out.append(start)
            # Now find the end.
            while(True):
                end = ends[i]
                # Is the next start within tmin_gap_raw from the current end?
                # If so, continue searching.
                # There is one exception to it: We're already at the end.
                # If we reach the last entry stop.
                if i + 1 >= max_i:
                    ends_out.append(end)
                    break    
                
                if end + tmin_gap_raw > starts[i+1]:
                    i += 1
                else:
                    ends_out.append(end)
                    break
        i += 1
        if i >= max_i:
            break
    
    return starts_out, ends_out        
        
        