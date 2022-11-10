'''
    File name: RS_file.py
    Author: Simon Waid
    Date created: 2021
    Python Version: 3.8, 3.9
'''


import xml.etree.ElementTree as ET
import os
import logging
import numpy as np
import pandas as pd
import binascii
from numba import njit 

# Fraction of chunks used for standard deviation and average calculation.
AVG_NO_CHUNKS=0.1

@njit
def _chunk_processor(data_in, chunk_size):
    
    no_chunks=int(len(data_in)/chunk_size)

    chunks_std=np.empty(no_chunks-1)
    chunks_avg=np.empty(no_chunks-1)
    for i in range(no_chunks-1):
        data=data_in[i*chunk_size:(i+1)*chunk_size]
        chunks_avg[i]=np.average(data)
        chunks_std[i]=np.std(data)
    
    return chunks_avg, chunks_std

class Burst_Indices():
    
    def __init__(self, starts, ends):
        '''
        Class for packing the starts and ends of a burst into a single object.
        '''
        self.starts=np.array(starts)
        self.ends=np.array(ends)
    
    def __len__(self):
        '''
        '''
        return len(self.starts)
    
    def chop_signal(self, data=None):
        '''
        Chops the provided signal using the saved start and end indices. 
        
        :returns: list of objects similar to data.
        '''
        
        result=[]
        for start, end in zip(self.starts, self.ends):
            result.append(data[start:end])
            
        return result
    
    def get_durations(self):
        '''
        Returns the difference between ends and starts
        '''
        
        durations=self.ends-self.starts
        
        return durations
    
    def filter_bool(self, filter):
        '''
        Returns a new instance of this class containing only starts and ends where the filter is True
        
        :param filter: Array of the same length as starts and ends
        '''
        
        starts= self.starts[filter]
        ends= self.ends[filter]
    
        new_class=Burst_Indices(starts, ends)
        
        return new_class

    
    def filter(self, min_idx, max_idx):
        '''
        Returns a new instance of this class containing only indices between min_idx and max_idx.
        
        :param min_idx:
        :param max_idx: 
        '''
        
        starts_fitler=(self.starts > min_idx) & (self.starts < max_idx)  
        ends_fitler=(self.ends > min_idx) & (self.ends < max_idx)
        
        starts=self.starts[starts_fitler]
        ends=self.ends[ends_fitler]
        
        new_class=Burst_Indices(starts, ends)
        
        return new_class
        
    def offset(self, offset):
        '''
        Adds the given offset to both starts and ends
        '''
        self.starts += offset
        self.ends += offset
        
        
    

def detect_bursts(data_in, chunk_size, min_len_pause, min_len_burst, burst_buffer, noise_dist = 7, max_dist = 0, return_cmp=False):
    '''
    Detect spills. 
    
    Concept:
        1. Find a pause between bursts to extract dark counts: 
            1.2. Divide data into chunks of size chunk_size. 
            1.3. For each chunk we compute average and standard deviation. When a signal is present, the standard deviation is large than when no signal is present.
            1.4. We search for the chunks with smallest standard deviation. These chunks should represent the background noise. 
        2. Find the beginning and end of bursts:
            2.1. Based on the data from 1.4 we calculate a threshold above which we assume a signal is present. 
            2.2. We use this threshold search for potential beginnings and ends of bursts. 
            2.3. We merge potential beginnings and ends. Between the end and the beginning of the next bursts there is a pause of at least min_len_pause.
            2.4. We add some buffer to the burst so our data will include the very beginning the burst.
    
    :param data_in: Linear array containing an intensity signal.
    :param chunk_size: Processing happens in chunks. Set this to approx. 20% of the minimum pause you expect between spills. Unit: number of samples. 
    :param min_len_pause: The minimum pause you expect between spills. Unit: number of samples.
    :param min_len_burst: The minimum length of a spill. Shorter spills will be ignored. Unit: number of samples.
    :param burst_buffer: The buffer that should be added at the beginning and end of each spill. Unit: number of samples.
    :param noise_dist: Optional.
    :param max_dist: Optional.
    :param return_cmp: Optional. Defaults to False. If True cmp will be returned, if False, cmp will be None.
    
    :returns  starts, ends, cmp, info: starts and ends contains the indexes at which spills start and end respectively. cmp and info contain internal information for debugging.
    
    '''
    
    
    # Only integer indices make sense. Convert the input if the user has not done so yet.
    chunk_size=int(chunk_size)
    min_pause=int(min_len_pause)
    min_spill=int(min_len_burst)
    burst_buffer=int(burst_buffer)
    # 1.2. and 1.3 calculate standard deviation and average for each chunk.
    
    chunks_avg, chunks_std=_chunk_processor(data_in, chunk_size)
    
    # 1.4, 1.5, 2.1 find the minimum standard deviation and calculate threshold
   
    # Getting a valid average and standard does not work well when using a single chunk.
    # Therefore we sort the chunks by std and evaluate the chunks with the lowest std.
    avg_no_chunks=int(len(chunks_avg)*AVG_NO_CHUNKS)
    chunks_std_idx_sort=np.argsort(chunks_std)
    chunks_std_idx=chunks_std_idx_sort[:avg_no_chunks]
    
    ref_avg=np.mean(chunks_avg[chunks_std_idx])
    ref_std_min=np.mean(chunks_std[chunks_std_idx])
    ref_std_min_th=ref_std_min*noise_dist
    
    # Using the noise as reference is not always reliable. We also have an alternative method.
    ref_chunk_idx_max=np.argmax(chunks_std)
    
    ref_std_max_th=chunks_std[ref_chunk_idx_max]*max_dist
    std_th=np.max([ref_std_min_th, ref_std_max_th])

    threshold_upper=ref_avg + std_th 
    threshold_lower=ref_avg - std_th 
    
    # 2.2 find potential beginnings and ends of spills   
    cmp_upper= np.array(data_in > threshold_upper, dtype=np.int8) 
    cmp_lower= np.array(data_in < threshold_lower, dtype=np.int8)
    cmp_a = np.logical_or(cmp_upper, cmp_lower).astype(np.int8)
    del cmp_upper 
    del cmp_lower
    cmp_b=cmp_a[:-1]-cmp_a[1:]
    ends=np.where(cmp_b == 1)[0]
    starts=np.where(cmp_b == -1)[0]
    del cmp_b
    
    # Handle the situation where we start or end with beam being present.
    if cmp_a[0]:
        #ends=np.insert(ends,0,0)
        starts=np.insert(starts,0,0)
        
    if cmp_a[-1]:
        #starts=np.append(starts,len(data_in)-1)
        ends=np.append(ends,len(data_in)-1)
        
    # 2.3 Merge potential spills 
    max_dist_pause=int(min_pause)
    
    dist=np.subtract(starts[1:], ends[:-1])
    real_spills=np.where(dist>max_dist_pause)[0]
    starts_real=starts[real_spills +1 ]
    ends_real=ends[real_spills]

    #
    if starts_real[0] > ends_real[0] :
        starts_real=np.insert(starts_real, 0, starts[0])
        ends_real=np.append(ends_real, ends[-1])
        
    # Eliminate spills which are too short
    dist=np.subtract(ends_real, starts_real)
    real_spills=np.where(dist>min_spill)[0]
    starts_real=starts_real[real_spills]
    ends_real=ends_real[real_spills]
        
    # 2.4 add buffer
    starts_buf=np.array([s - burst_buffer if s - burst_buffer > 0 else 0  for s in starts_real]) 
    ends_buf=np.array([s + burst_buffer if s + burst_buffer < len(data_in) else len(data_in)-1 for s in ends_real])
    
    # Pack the indices into a class
    indices_buf=Burst_Indices(starts_buf, ends_buf)
    indices_raw=Burst_Indices(starts, ends)
    indices_nobuf=Burst_Indices(starts_real, ends_real)
    
    
    info={'chunks_std_idx': chunks_std_idx,\
          'chunks_std': chunks_std,\
          'chunks_avg': chunks_avg, \
          'ref_avg' : ref_avg, \
          'threshold_upper': threshold_upper, \
          'threshold_lower': threshold_lower, \
          'indices_raw': indices_raw, \
          'indices_nobuf': indices_nobuf}
    
    # The content of cmp has the same size as data_in and can thus become quite large. 
    # As it is only used for debugging, we normally don't care about it and thus avoid the overhead of returning it. 
    if not return_cmp:
        cmp_a=None
    
    return indices_buf, cmp_a, info 

class RS_File():
    '''
    Reads Rhode & Schwarz RTP16 files. Currently limited to .csv and .bin files. The oscilloscope writes two files. 
    In case of bin files this is a *.bin and a *.Wfm.bin file. The first once contains metadata. 
    The second the measurement data. Both files are needed. 
    They must be in the same directory and follow the R&S naming convention.
        
    The file format is described here: https://www.rohde-schwarz.com/webhelp/RTP_HTML_UserManual_en/Content/42894a16e1e94a45.htm
       
    Currently only normal waveform data is supported. Other data will be rejected. 
        
    :property xml_tags: Dictionary. The keys are the 'Name' attributes and the 'Values' attributes from the .xml file. This is a convenient way of accessing the most interesting part of the xml file. 
    :property xml_root: The metadata from the .xml file. This an instance of .. .
    :property meta: Dictionary containing meta data. 
    
    '''
    # We work with a lot of data. Theoretically using vaex would be a good idea.
    # Practically vaex does not work with 32 bit floats and plotting with PyQt and vaex is a pain.
    # So for the moment vaex is an option that is disabled. 
        
    USE_VAEX = False
    if USE_VAEX: 
        DTYPE_OUT = np.float64
    else:
        DTYPE_OUT = np.float32
    
    def __init__(self, file_name):
        '''
        :param file_name: The file name without extension.
        '''
        # Read the metadata first. We provide the data to the user via the .xml property.
        # If the data is anything else than a normal waveform (TraceType is eRS_TRACE_TYPE_NORMAL) we reject the file.
        
        file_base, file_extension = os.path.splitext(file_name)
        self.metadata_file = file_name
        self.waveform_file = file_base + '.Wfm' + file_extension
        # Now decode the file. 
        # If this takes too long one should consider not executing this during the class initialization.
        if file_extension == '.bin':
            _, self.xml_tags = self._loadXml(self.metadata_file)
            logging.info(f'Loaded xml file {self.metadata_file}.')
            self._data = None
            self.meta = self._decodeBin(self.xml_tags)
        elif file_extension == '.csv':
            logging.info(f'Processing {self.metadata_file}.')
            self.csv_meta = self._loadCsvMeta(self.metadata_file)
            self._data, self.meta=self._decodeCsvWfm(self.waveform_file, self.csv_meta)
        else:
            raise(RuntimeError('File type not implemented.'))


        self.bursts={}
        self.meta['metadata_file'] = self.metadata_file

    def _genTimeAxis(self, acquistion, start, stop):
        '''
        Generate a time axis for the given sample interval
        
        :param start: Start of the interval as sample number
        :param stop: End of the interval as sample number
        '''
        
        # Things get easier if we have proper numbers as start and end 
        if start is None:
            start =0
        if stop is None:
            if acquistion is None:
                stop = self.total_no_samples 
            else:
                stop = self.samples_per_acquisition    
                
        # Calculate start and stop in the time domain
        xStart = self.meta['xStart'] + start*self.meta['t_sample']
        xStop = self.meta['xStart'] + stop*self.meta['t_sample']
            
        length = stop - start
        #t=time.time()
        
        timestamps = np.linspace(xStart, xStop, length, dtype = np.float64)
        return timestamps
    
    def _loadXml(self, xml_file):
        '''
        Load the xml meta data
        '''
        xml_root = ET.parse(xml_file).getroot()
        xml_tags= {}
        # Iterate over all elements in the xml tree and process the attribues.
        # At the end we generate a flat dictionary with all attributes of the xml tree disregarding the hirarchy.
         
        for elem in xml_root.iter():
            if 'Name' in elem.attrib:
                k=elem.attrib['Name']
                if k in xml_tags:
                    # We don't overwrite keys in our flattened metadata structure. 
                    # If duplicate keys would appear we fail.
                    raise(RuntimeError('Key already exists. This is unexpected. Ciao!'))
                else:
                    xml_tags[k] = elem.attrib
        
        return xml_root, xml_tags

    def forgetRaw(self):
        self._data = None
        
    def detect_bursts(self, source, chunk_size, min_len_pause, min_len_burst, burst_buffer, noise_dist = 7, max_dist = 0):
        '''
        Detects bursts in the signal and returns the indices of the beginning and end of the bursts. The signal is split into chunks.
        Note: the result of the function is cached in memory.
        
        :param source: Name of a single channel to be analyzed. TODO: Make this consistent with all other functions.
        :param chunk_size:  See :py:`meth:detect_bursts`
        :param min_pause:  See :py:`meth:detect_bursts`
        :param min_spill:  See :py:`meth:detect_bursts`
        :param spill_buffer:  See :py:`meth:detect_bursts`
        
        '''
        
        key=f'{source}, {chunk_size}, {min_len_pause}, {min_len_burst}, {burst_buffer}, {noise_dist}, {max_dist}'
        
        if not key in self.bursts:
            
            data_in=self.getRaw(sources=source)
            self.bursts[key]=detect_bursts(data_in, chunk_size, min_len_pause, min_len_burst, burst_buffer, noise_dist = noise_dist, max_dist = max_dist, return_cmp=False)
        
        return self.bursts[key]
        

    def _decodeCsvWfm(self, file_name, csv_meta):
        '''
        Decodes csv data. 
        '''
        #noChannels=int(csv_meta['MultiChannelSource'][0])
        col_names=[]
        # It seems we always have to handle single and multiple channels separately. 
        if csv_meta['SignalFormat'] == 'XYDOUBLEFLOAT':
            col_names.append('Time')
        if csv_meta['MultiChannelExport'][0] == 'Off':
            col_names.append(csv_meta['Source'][0])
        elif csv_meta['MultiChannelExport'][0] == 'On':
            for name in csv_meta['MultiChannelSource'][1:]: 
                # It seems we can get a channel listed twice but only once in the data.
                # So if a channel is listed multiple times we ingore it :(.
                if not name == 'None' and not name in col_names:
                    col_names.append(name)
        else:
            raise(RuntimeError('This is a bug!'))
        
        # Load the waveform data
        csv_data=pd.read_csv(file_name, names=col_names, header=None)
        
        meta={}
        
        meta['length_total'] = len(csv_data)
        meta['length_acquisition'] = int(csv_meta['RecordLength'][0])
        meta['xStart'] = float(csv_meta['XStart'][0])
        meta['xStop'] = float(csv_meta['XStop'][0])
        meta['t_sample'] = (meta['xStop']-meta['xStart'])/meta['length_acquisition']
        # This attribute does not exist for csv data 
        meta['leading_samples']=0
        meta['data_format'] = 'pandas'
        # CSV data never needs the application of an offset or scaling factor
        meta['apply_offset'] = False
        
        meta['source_names'] =col_names
        meta['channel_type'] = []
        if csv_meta['SignalFormat'] == 'XYDOUBLEFLOAT':
            meta['channel_type'] = 'X-Time'
        for c in col_names:
            meta['channel_type'].append('Y')
        
        # Match  the data format of binary files.
        csv_data=csv_data.to_dict(orient='series')
        # Get rid of the Series and get numpy arrays instead
        csv_data={k:v.values for (k,v) in csv_data.items() }
        return csv_data, meta 
    
    def _loadCsvMeta(self, file_name):
        '''
        Loads the csv meta data into memory. 
        '''
        metadata = {}
        with open(file_name, 'r') as f:            
            for line in f:
                #line=line.replace('\n', '')
                m=line.split(':')
                if m[0] in metadata:
                    raise(RuntimeError('Bug!'))
                else: 
                    metadata[m[0]] = m[1:-1]
        return metadata
        
    def _decodeBin(self, xml_tags):
        '''
        Decodes binary data and stores the metadata in memory. 
        '''
        # There seem to be many different input formats.
        # In the following we assemble the format for numpy.
        # We're defining the options for decoding the binary data here.
        # Note: There are many parameters which might have an impact.
        # It is rather obscure which ones do and how. Consequently,
        # for the moment we simply assert certain configuration parameters.
        # It is not clear what happens if the 'UserValue' attribute contradicts the 'Value' attribute. 
        # It seems that typically the 'Value' attribute takes precedence.
        
        # Asserting some User values. This might not be needed.
        logging.debug(f'UserValue SignalFormat: {xml_tags["SignalFormat"]}')
        
        logging.debug(f'UserValue ByteOrder: {xml_tags["ByteOrder"]}')

        # Asserting the byte order as we don't have any samples with another byte oder.
        # Perhaps there is a bug in the oscilloscope. The byte order does not seem to change.
        # This might cause the one or the other surprise in the future.
        if xml_tags['ByteOrder']['Value'] == 'eRS_BYTE_ORDER_MSB_FIRST':
            byte_order='<'
        elif xml_tags['ByteOrder']['Value'] == 'eRS_BYTE_ORDER_LSB_FIRST':
            byte_order='<'
        else:
            raise(RuntimeError(f"Invalid ByteOrder {xml_tags['ByteOrder']['Value']}"))
        
        # Handle single/multi channel formats. 
        # We need the number of columns in the data and their names. 
        source_names = []
        if xml_tags["MultiChannelExport"]['Value']=='eRS_ONOFF_OFF':
            no_columns = 1
            source_names.append(xml_tags["Source"]['Value'])
            #channel_names.append(xml_tags["Source"]['Value'])
        elif xml_tags["MultiChannelExport"]['Value']=='eRS_ONOFF_ON':
            channel_names = []
            no_columns = 0
            for ch in ['I_0', 'I_1', 'I_2', 'I_3']:
                if self.xml_tags["MultiChannelExportState"][ch]=='eRS_ONOFF_ON':
                    source_names.append(self.xml_tags["MultiChannelSource"][ch])
                    channel_names.append(ch)
                    no_columns+=1
        else:
            raise(RuntimeError('Unknown value for MultiChannelExport.'))
        
        meta = {}
        meta['length_raw_expected'] = int(xml_tags['SignalHardwareRecordLength']['Value']) 
        meta['leading_samples'] = int(xml_tags['LeadingSettlingSamples']['Value'])

        # Shorten the column names. We don't need any eRS etc.
        source_names = [x.replace('eRS_SIGNAL_SOURCE_','') for x in source_names]
        meta['source_names']=source_names
        # Convert the data format into something numpy can use.
        # Also determine if the offset and position shall be applied
        encoding_format=[]
        meta['channel_type']=[]
        
        # Enable calculation of the expected file size 
        bytes_per_sample=0
        if xml_tags['SignalFormat']['Value'] == 'eRS_SIGNAL_FORMAT_XYDOUBLEFLOAT':
           
            # There seems to be a bug in the oscilloscope. The byte order is wrong for this data type.
            byte_order='<'
            encoding_format.append(('Time', byte_order+'f8'))
            if xml_tags['XAxisTDRDomain']['Value'] == 'eRS_TDR_TDT_RESULT_SIGNAL_DOMAIN_TIME':
                # Store the type of data in the wfm file 
                meta['channel_type'].append('X-Time')
                bytes_per_sample+=8
            else:
                raise(RuntimeError('X-axis domain not implemented'))
            for ch in source_names:
                bytes_per_sample+=4
                encoding_format.append((ch, byte_order+'f4'))
                meta['channel_type'].append('Y')
            meta['apply_offset']=False
        elif xml_tags['SignalFormat']['Value'] == 'eRS_SIGNAL_FORMAT_FLOAT':
            for ch in source_names:
                bytes_per_sample+=4
                encoding_format.append((ch, byte_order+'f4'))
                meta['channel_type'].append('Y')
            meta['apply_offset']=False
        elif xml_tags['SignalFormat']['Value'] == 'eRS_SIGNAL_FORMAT_INT8BIT':
            for ch in source_names:
                bytes_per_sample+=1
                encoding_format.append((ch, byte_order+'i1'))
                meta['channel_type'].append('Y')
            meta['apply_offset'] = True
        elif xml_tags['SignalFormat']['Value']=='eRS_SIGNAL_FORMAT_INT16BIT':
            for ch in source_names:
                bytes_per_sample+=2
                encoding_format.append((ch, byte_order+'i2'))
                meta['channel_type'].append('Y')
                #quantisation_levels=int(xml_tags['NofQuantisationLevels']['Value']) 
            meta['apply_offset'] = True
        else:
            raise(RuntimeError(f"File format (SignalFormat) is not supported.\n \
                                 Tag: {xml_tags['SignalFormat']}\n \
                                 XML file: {self.metadata_file} "))

        meta['quantisation_levels']= quantisation_levels= int(xml_tags['NofQuantisationLevels']['Value'])
        meta['encoding_format'] = encoding_format
        meta['channel_name'] = [x[0] for x in encoding_format]
        # Ensure the data format is normal. In the future other data formats might need separate hanlding.
        if self.xml_tags["TraceType"]['Value'] == 'eRS_TRACE_TYPE_NORMAL':
            pass
        elif self.xml_tags["TraceType"]['Value'] == 'eRS_TRACE_TYPE_AVERAGE':
            pass
        else:
            raise(RuntimeError(f'File format {self.xml_tags["TraceType"]["Value"]} is not supported. Have fun implementing it :)!'))

        # Output some info to the log.
        logging.info(f'RecordLength: {xml_tags["RecordLength"]}') 
        logging.info(f'TriggerOffset: {xml_tags["TriggerOffset"]}') 
        logging.info(f'SignalRecordLength: {xml_tags["SignalRecordLength"]}') 
        logging.info(f'Source: {xml_tags["Source"]}')
        
        # Save the conversion factor and offset for later processing
        meta['conversion_factor'] = {}
        meta['offset'] = {}
        if meta['apply_offset']:
            # Single and multi-channel needs seaprate handling it seems.
            if xml_tags["MultiChannelExport"]['Value'] =='eRS_ONOFF_OFF':
                # The Position is indicated in divisions. We have to convert to V
                position_div = float(xml_tags['VerticalPosition']['Value'])
                verticalScale = float(xml_tags['VerticalScale']['Value'])
                step_factor = float(xml_tags['VerticalScale']['StepFactor'])
                position = position_div * verticalScale
                # The offset is in V.
                offset = float(xml_tags['VerticalOffset']['Value'])
                meta['conversion_factor'][source_names[-1]] = 1 / float(quantisation_levels) * step_factor * verticalScale        
                meta['offset'][source_names[-1]] = offset - position
            elif xml_tags["MultiChannelExport"]['Value'] == 'eRS_ONOFF_ON':
                for sn, ch in zip(source_names, channel_names):
                    position_div = float(xml_tags['MultiChannelVerticalPosition'][ch])
                    verticalScale = float(xml_tags['MultiChannelVerticalScale'][ch])
                    step_factor = float(xml_tags['MultiChannelVerticalScale']['StepFactor'])
                    position = position_div * verticalScale
                    offset = float(xml_tags['MultiChannelVerticalOffset'][ch])
                    meta['conversion_factor'][sn] = 1 / float(quantisation_levels) * step_factor * verticalScale        
                    meta['offset'][sn] = offset - position
        
        # Info for fast segmentation
        meta['NumberOfAcquisitions']=int(xml_tags['NumberOfAcquisitions']['Value'])
        
        # Save some metadata
        meta['xStart'] = float(xml_tags['XStart']['Value'])
        meta['xStop'] = float(xml_tags['XStop']['Value'])
        meta['length_total'] = int(xml_tags['RecordLength']['Value'])*meta['NumberOfAcquisitions']
        meta['length_acquisition'] = int(xml_tags['RecordLength']['Value'])
        meta['t_sample'] = (meta['xStop']-meta['xStart'])/meta['length_acquisition']
        meta['data_format']='numpy'
        
        meta['bytes_per_sample']=bytes_per_sample
        # Bytes at the beginning of the file which do not store data. This is a magic number.
        meta['initial_offset']=8
        # We don't have this information yet, so initialize the value.
        meta['acquistions_incomplete'] = None
        return meta

    def timeToIndex(self, time, out_of_range="None"):
        '''
        Returns an index corresponding to the given time in seconds. 
        
        :param time: Time in seconds
        :out_of_range: Controls the way out of range values are handled. Valid options are "None" and "Nearest".\ 
        "None" will return None for out of range values of time. "Nearest" will return the closest valid index.  Optional, defaults to "None". 
        '''
        
        if time < self.meta['xStart']:
            if out_of_range == "None":
                return None
            elif out_of_range == "Nearest":
                return self.meta['xStart']
            else:
                raise(RuntimeError(f'Invalid value for out_out_range: {out_of_range}'))
            
        if time >  self.meta['xStop']:
            if out_of_range == "None":
                return None
            elif out_of_range == "Nearest":
                return self.meta['xStop']
            else:
                raise(RuntimeError(f'Invalid value for out_out_range: {out_of_range}'))
            
        time_offset=time-self.meta['xStart']
        index=time_offset/self.meta['t_sample']
        return int(index)


    def indexToTime(self, index, out_of_range="None"):
        '''
        Returns the time corresponding to the given index in seconds. Returns None if the time is out of range.

        :param index: Index value
        :out_of_range: Controls the way out of range values are handled. Valid options are "None" and "Nearest".\ 
        "None" will return None for out of range values of time. "Nearest" will return the closest valid index.  Optional, defaults to "None". 
        
        '''
        if index < 0:
            if out_of_range == "None":
                return None
            elif out_of_range == "Nearest":
                return 0
            else:
                raise(RuntimeError(f'Invalid value for out_out_range: {out_of_range}'))
            
        if index > self.meta['length_total']:
            if out_of_range == "None":
                return None
            elif out_of_range == "Nearest":
                return self.meta['length_total']
            else:
                raise(RuntimeError(f'Invalid value for out_out_range: {out_of_range}'))
        
        time_offset=self.meta['xStart']
        time=index*self.meta['t_sample']+time_offset
        return time

    def check_file(self):
        '''
        Performs some checks on the files. This includes comparing the file size of the binary file with the expected file size calculated using the 
        meta data and trying to memmap the file including sanity checks on the memmap. 
        
        :returns: size_matches, acquistions_incomplete, file_corrupted  
        
        size_matches: True if the file size matches, False if they differ.
        acquistions_incomplete: None if the file cannot be read. True if a multiple acquisitions are expected but not all are present in the file. 
        file_corrupted: True if memmapping the file and corresponding sanity checks fail. This may also indicate an unsupported file format.
        '''
        
        # Try to read the file first. This will correct the metadata for files which don't contain the complete history  
        # Reading will fail on corrupted files, therefore we ignore any failure to do so.
        file_corrupted=False
        try:
            self.getRaw()
        except:
            file_corrupted=True
        no_samples_in_file=self.meta['NumberOfAcquisitions']*self.meta['length_raw_expected']
        expected_file_size=self.meta['initial_offset']+no_samples_in_file*self.meta['bytes_per_sample']
        file_size_on_disk=os.path.getsize(self.waveform_file)
        print(f'Expected file size: {expected_file_size}, File size on diks: {file_size_on_disk}')
        
        sizes_match=expected_file_size == file_size_on_disk
        acquistions_incomplete=self.meta['acquistions_incomplete']
        
        return sizes_match, acquistions_incomplete, file_corrupted

    def check_xml(self):
        '''
        Does a simple sanity check on the xml file. A correctly written xml file must end with "</Database>".
        '''
        with open(self.metadata_file) as f:
            for line in f:
                pass
            last_line = line
        
        print(last_line)
        
        return last_line=='</Database>'

    @property
    def data_raw_RS(self):
        if self._data is None:
        # Load the data via numpy. To get a numpy array we first load as a single array.
        # Then we reshape and convert to float as we want to do some math on the data later on.
        # We seem to have 8 bytes of offset
        # Just load the data once it is really needed, not already during initializing.
            
            
            #with open(self.waveform_file,"rb") as file:
            #    self.binary_header_a=file.read(4)
            
            self.binary_header=np.fromfile(self.waveform_file, dtype='<i4', count=2, offset=0)
            #self.binary_header_b=np.fromfile(self.waveform_file, dtype='<i4', count=1, offset=4)
            
            #self.size=2**self.binary_header[0] * self.binary_header[1] 
            #print('Binary header A: {}'.format(hex(self.binary_header_a[0])))
            print('Binary header: {}, HW length: {}'.format(self.binary_header, self.meta['length_raw_expected']))
            #print('Binary multiply: {}'.format(self.binary_header[0]))
            #print('Size: {}'.format(self.size))
            print('Encoding format: {}'.format(self.meta['encoding_format']))
            
            #print('Binary header B: {}'.format(self.binary_header_b))
            #print('Binary header')
            #if not self.binary_header == b'\x04\x00\x00\x00\xe6\x0f\x00\x00':
            #    raise(RuntimeError(f'Found unexpected header. {self.metadata_file }: {self.binary_header}'))
            
            # Get the data from the file
            self.memmap = np.memmap(self.waveform_file, dtype=self.meta['encoding_format'], offset = self.meta['initial_offset'], mode='r')

            # Reshape.
            # We keep the reshaped data in memory for fast access later on.
            self._data={}
            for n in self.meta['channel_name']:
                self._data[n]=self.memmap[n]
            
            # Do some sanity check. We expect the 'Value' of "SignalHardwareRecordLength" of raw samples. 
            # Note1: At this point we expect to see the 'Value' and not the 'UserValue'.

            length_raw_expected = self.meta['length_raw_expected']*self.meta['NumberOfAcquisitions']
            length_raw_observed = len(self._data[self.meta['channel_name'][0]])
            
            # Doing sanity checks here is tricky.
            # 1. There seems to be a bug in the oscilloscope software. For xy data the length does not match even if the data is fine
            # 2. If NumberOfAcquisitions is larger than 1 this seems to be an upper bound and not the actual lenght of the acquisitions.
           
            self.meta['acquistions_incomplete']=False
            if length_raw_expected != length_raw_observed: # and not self.meta['ignore_lengthcheck']:
                message=f'Sanity check failed. SignalHardwareRecordLength does not match decoded data. \n length_raw_expected:{length_raw_expected}, length_raw_observed:{length_raw_observed}'
                if self.meta['NumberOfAcquisitions'] > 1:
                    if length_raw_expected > length_raw_observed and length_raw_observed%self.meta['length_raw_expected'] == 0:
                        # We ignore this case. It happens if the incomplete history is saved.
                        self.meta['length_total']=len(self.memmap)
                        self.meta['NumberOfAcquisitions']=self.meta['length_total']/self.meta['length_raw_expected']
                        self.meta['acquistions_incomplete']=True
                    else:
                        raise(RuntimeError(message))
                else:
                    raise(RuntimeError(message))
                        
            logging.info(f'Loading binary data from file {self.waveform_file} completed.')

        return self._data

    def rawToDtypeOut(self, data, source=None):
        '''
        Helper function converting the data to DTYPE_OUT and applying the conversion_factor, offset and scaling to the given data.        
        
        :param data: dict like self.data_raw or numpy array
        :param sensor: Must be given if data is a numpy array. 
        '''
        
        if type(data) == dict:
            if self.meta['apply_offset']:
                for key in data.keys():
                    data[key]=data[key].astype(self.DTYPE_OUT)
                    data[key] *= self.meta['conversion_factor'][key]
                    data[key] += self.meta['offset'][key]
        else:
            if not source is None:
                data=data.astype(self.DTYPE_OUT)
                data *= self.meta['conversion_factor'][source]
                data += self.meta['offset'][source]
            else:
                raise(RuntimeError('Wrong parameters'))
    
        return data

    def _gen_start_stop_idx(self, acquisition, start, stop ):
        '''
        Generate start stop indices for our memory mapped data.
        
        '''
        if not start is None and not stop is None:
            if stop <= start:
                raise(RuntimeError('stop must be larger than start'))
        
        # If the data includes multiple acquisitions, the user may restrict the data access to one acquisitions
        # Alternatively, the complete data  
        if acquisition is None:
           
            # Sanity check on start and stop
            if not stop is None:
                if stop < 0 or stop > self.samples_per_acquisition:
                    raise(RuntimeError(f'Invalid value for stop {stop}')) 
            if not start is None:
                if start < 0 or start > self.samples_per_acquisition:
                    raise(RuntimeError(f'Invalid value for stop {start}')) 
           
            # Compute start and stop indices cutting away the leading samples.  
            start_idx=self.meta['leading_samples']
            if stop is None:
                stop_idx = start_idx + self.meta['length_total']
            else:
                stop_idx = start_idx + stop  
            
            if not start is None:
                start_idx += start 
        else: 
            
            # Sanity check on start and stop
            if not stop is None:
                if stop < 0 or stop > self.meta['length_total']:
                    raise(RuntimeError(f'Invalid value for stop {stop}')) 
           
            if not start is None:
                if start < 0 or start > self.meta['length_total']:
                    raise(RuntimeError(f'Invalid value for stop {start}')) 
        
            # Sanity check on the acquisition number
            if acquisition < 0 or acquisition > self.no_acquisitions:
                raise(RuntimeError(f'Invalid  acquisition number: {acquisition}'))
            
            # Do some sanity check. I have never encountered LeadingSettlingSamples unequal 0 when  NumberOfAcquisitions was unequal to zero so I don't know how to handle this situation.
            if self.meta['leading_samples'] != 0 :
                raise(RuntimeError("Encountered LeadingSettlingSamples != 0 while trying to access a single acquisition. It's unclear how to handle this. Therefore I'm giving up")) 
            
            # Compute start and stop indices
            start_idx = acquisition*self.samples_per_acquisition
            if stop is None:
                stop_idx = start_idx + self.samples_per_acquisition 
            else:
                stop_idx = start_idx + stop
            
            if not start is None:
                start_idx += start

        return start_idx, stop_idx

    def _sanitize_sources(self, sources):
        '''
        Ensures the provided sources are a list. If None is provided a list of all available sources is returned. 
        If a string is provided instead a list, a list is created from the string. 
         
        '''
    
        if sources is None:
            sources= self.meta['source_names']

        if type(sources) == str:
            sources=[sources]
        
        return sources

    def getRaw(self, acquisition=None, start = None, stop = None, sources=None):
        '''
        Filters the raw data using start stop and source.
        Returns something similar to the raw data but filtered. Time data is removed.
        
        :param acquisition: 
        :param start: Index of starting sample
        :param stop: Index of final sample
        :param sources: Oscilloscope Source Names. Optional, defaults to None. If None all channels will be selected. Can also be a list or string pointing to a single channel.  
        
        :returns: If sources was None or a list a dictionary is returned. If sources was a string the data for that channel is returned.
        '''
        
        sources_sane=self._sanitize_sources(sources)
        
        # A bit of cryptic python code:). 
        # my_data is a dict like self.data_raw but filtered to source or meta['source_names']. 
        # This way we eliminate the time.
        #if sources is None:
        #    my_data={k:v for (k,v) in self.data_raw_RS.items() if k in self.meta['source_names']}
        #else:
        my_data={k:v for (k,v) in self.data_raw_RS.items() if k in sources_sane}
        
        start_idx, stop_idx =self._gen_start_stop_idx(acquisition, start, stop)
           
        # Now apply the start stop filter
        # We will handle the time separately
        for key in my_data.keys():
            my_data[key]=my_data[key][start_idx:stop_idx]
         
        # If we got a single string as an input, don't return a dictionary but the requested single channel. 
        if type(sources) == str:
            return my_data[sources]
         
        return my_data
    
    @property
    def sample_time(self):
        '''
        Duration of one sample in seconds
        '''
        return self.meta['t_sample']
        
    @property
    def all_channels(self):
        '''
        Lists all channels in the data including a time channel if available
        '''
        return self.meta['channel_name']
    
    @property
    def signal_sources(self):
        '''
        List of signal sources
        '''
        sources=self.meta['source_names']
        return sources
    
    @property
    def no_acquisitions(self):
        '''
        Number of acquisition e.g. in case a history was saved
        '''
        return self.meta['NumberOfAcquisitions']
        
    @property
    def samples_per_acquisition(self):
        '''
        Number of samples per acquisition
        '''
        return self.meta['length_acquisition']
    
    @property
    def total_no_samples(self):
        '''
        Total number of samples
        '''
        return self.meta['length_total']
    
    def getAsDf(self, acquisition=None, start = None, stop = None, sources=None, time=True):
        '''
        Returns a dataframe containing the data between start and stop. By default, also timestamps will be included.
        You can set time to False to exclude timestamps.
        This is useful for plotting or other processing of large files.
        
        :param start: Index of starting sample
        :param stop: Index of final sample
        :param source: Oscilloscope Source Name 
        :param time: Optional. Defaults to True. If false, no time data is returned.
        :type time: Boolean
        '''
        # Filtering is done a separate function. We get filtered data without time.
        my_data=self.getRaw(acquisition=acquisition, start = start, stop = stop, sources=sources)
        
        # We don't have time information in our data yet.
        # It's time to convert and scale.
        my_data=self.rawToDtypeOut(my_data)
        
        # We now have to add the time if requested by the user.
        # If xy data was selected we get a time axis from the oscilloscope. 
        # Otherwise we have to generate it.
        if time:
            if  'X-Time' in self.meta['channel_type']:
                tsChannel=self.meta['channel_type'].index('X-Time')
                name=self.meta['channel_name'][tsChannel]
                # If we have x-y data we also have to apply the start stop filter.
                start_idx, stop_idx=self._gen_start_stop_idx(acquisition, start, stop)
                my_data[name]=self.data_raw_RS[name][start_idx:stop_idx]
            else:
                # Generate time stamps. We only generate them for the inteval between start and stop
                try:
                    timestamps=self._genTimeAxis(acquisition, start, stop)
                except:
                    print('Breakpoint')
                    raise
                my_data['Time']=timestamps
        
        # Generate a DataFrame
        try:
            df = pd.DataFrame(my_data)
        except:
            print('Breakpoint')
            raise
        
        return df

    def getLimitedStartStop(self):
        '''
        Returns start and stop value for :py:meth:`.getAsDf` or :py:meth:`.getRaw` with a limited length.
        The start stop values are centered to the 
        
        :returns: start, stop
        '''
        MAXPLOTLEN = 5E7
        
        dfLen = self.no_samples_acquisitions
        
        # If greater than the predefined length, one defines a region in the middle of the timeframe with
        # size MAXPLOTLENGTH
        if dfLen > MAXPLOTLEN:
            start = int(dfLen/2-MAXPLOTLEN/2)
            stop = int(dfLen/2+MAXPLOTLEN/2)

        # Unnecessary but doesn't make it faster anyway
        else:
            start = 0
            stop = int(dfLen)

        # Just in case: saturate
        if start < 0:
            start = 0
        if stop > dfLen:
            stop = dfLen

        
        return start, stop
    
