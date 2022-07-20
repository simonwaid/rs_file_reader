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
import numpy as np
import binascii

class RS_File():
    '''
    Reads Rhode & Schwarz RTP16 files. Currently limited to .csv and .bin files. The oscilloscope writes two files. 
    In case of bin files this is a *.bin and a *.Wfm.bin file. The first once contains metadata. 
    The second the measurement data. Both files are needed. 
    They must be in the same directory and follow the R&S naming convention.
        
    The file format is described here: https://www.rohde-schwarz.com/webhelp/RTP_HTML_UserManual_en/Content/42894a16e1e94a45.htm
       
    Currently only normal waveform data is supported. Other data will be rejected. 
        
    :property xml_tags: Dictionary. The keys are the 'Name' attribues and the 'Values' attributes from the xml file. This is a convenient way of accessing the most interesting part of the xml file. 
    :property xml_root: The metadata from the xml file. This an instance of .. .
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
            self.meta = self._decodeBinary(self.xml_tags)
        elif file_extension == '.csv':
            logging.info(f'Processing {self.metadata_file}.')
            self.csv_meta = self._loadCsvMeta(self.metadata_file)
            self._data, self.meta=self._decodeCsvWfm(self.waveform_file, self.csv_meta)
        else:
            raise(RuntimeError('File type not implemented.'))

        self.meta['metadata_file'] = self.metadata_file

    def _genTimeAxis(self, start, stop):
        '''
        Generate a time axis for the given sample inverval
        
        :param start: Start of the interval as sample number
        :param stop: End of the interval as sample number
        '''
        
        # Things get easier if we have proper numbers as start and end 
        if start is None:
            start =0
        if stop is None:
            stop = self.meta['length_total'] 
        
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
        
    def _decodeBinary(self, xml_tags):
        '''
        Decodes binary data and strores the metadata in memory. 
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
        meta['trailing_Sample'] = int(xml_tags['LeadingSettlingSamples']['Value'])

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
        quantisation_levels = int(xml_tags['NofQuantisationLevels']['Value'])
        meta['quantisation_levels']= quantisation_levels
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
        logging.info(f'RecordLength: {xml_tags["TraceType"]}') 
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
        meta['length_total'] = int(xml_tags['SignalRecordLength']['Value'])*meta['NumberOfAcquisitions']
        meta['length_acquisition'] = int(xml_tags['SignalRecordLength']['Value'])
        meta['t_sample'] = (meta['xStop']-meta['xStart'])/meta['length_acquisition']
        meta['data_format']='numpy'
        
        meta['bytes_per_sample']=bytes_per_sample
        # Bytes at the beginning of the file which do not store data. This is a magic number.
        meta['initial_offset']=8
        
        return meta

    def timeToIndex(self, time):
        '''
        Returns an index corresponding to the given time in seconds. Returns None if the time is out of range.
        '''
        
        if time < self.meta['xStart'] or time >  self.meta['xStop']:
            return None
        
        time_offset=time-self.meta['xStart']
        index=time_offset/self.meta['t_sample']
        return int(index)

    def check_file_size(self):
        '''
        Compares the file size of the binary file with the expected file size calculated using the 
        meta data. 
        
        :returns:  True if the file size matches, False if they differ.  
        '''
        
        self.getRaw()
        no_samples_in_file=self.meta['NumberOfAcquisitions']*self.meta['length_raw_expected']
        expected_file_size=self.meta['initial_offset']+no_samples_in_file*self.meta['bytes_per_sample']
        file_size_on_disk=os.path.getsize(self.waveform_file)
        print(f'Expected file size: {expected_file_size}, File size on diks: {file_size_on_disk}')
        
        sizes_match=expected_file_size == file_size_on_disk
        file_incomplete=self.meta['file_incomplete']
        
        return sizes_match, file_incomplete

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
            
            # Get the data from the file
            #self.meta['encoding_format']=[('CH2_TR1', '>i1')]
            self.memmap = np.memmap(self.waveform_file, dtype=self.meta['encoding_format'], offset = self.meta['initial_offset'], mode='r')
            #crc= binascii.crc32(self.memmap)

            #print('crc32 = {:#010x}'.format(crc))
            # Reshape
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
           
            self.meta['file_incomplete']=False
            if length_raw_expected != length_raw_observed: # and not self.meta['ignore_lengthcheck']:
                message=f'Sanity check failed. SignalHardwareRecordLength does not match decoded data. \n length_raw_expected:{length_raw_expected}, length_raw_observed:{length_raw_observed}'
                if self.meta['NumberOfAcquisitions'] > 1:
                    if length_raw_expected > length_raw_observed and length_raw_observed%self.meta['length_raw_expected'] == 0:
                        # We ignore this case. It happens if the incomplete history is saved.
                        self.meta['length_total']=len(self.memmap)
                        self.meta['file_incomplete']=True
                    else:
                        raise(RuntimeError(message))
                else:
                    raise(RuntimeError(message))
                    
            # Cut off leading and trailing samples. It seems the 'Value' of 'LeadingSettlingSamples' is what we should use.
            # Note1: Again in real data we observe the 'Value' and not the 'UserValue'.  
            trailing_Sample = self.meta['trailing_Sample']
            
            # Keep the data in memory for fast access later on.
            for key in self._data.keys():
                self._data[key] =  self.memmap[key][trailing_Sample:(trailing_Sample + self.meta['length_total'])]
    
            logging.info(f'Loading binary data from file {self.waveform_file} completed.')

        return self._data

    def rawToDtypeOut(self, data):
        '''
        Helper function converting the data to DTYPE_OUT and applying the conversion_factor, offset and scaling to the given data.        
        
        :param data: dict like self.data_raw
        '''
        
        if self.meta['apply_offset']:
            for key in data.keys():
                data[key]=data[key].astype(self.DTYPE_OUT)
                data[key] *= self.meta['conversion_factor'][key]
                data[key] += self.meta['offset'][key]
        
        return data

    def getRaw(self, start = None, stop = None, source=None,):
        '''
        Filters the raw data using start stop and source.
        Returns somehing similar to the raw data but filtered. Time data is removed.
        
        :param start: Index of starting sample
        :param stop: Index of final sample
        :param source: Oscilloscope Source Name 
        '''
        
        # A bit of cryptic python code:). 
        # my_data is a dict like self.data_raw but filtered to source or meta['source_names']. 
        # This way we eliminate the time.
        if source is None:
            my_data={k:v for (k,v) in self.data_raw_RS.items() if k in self.meta['source_names']}
        else:
            my_data={k:v for (k,v) in self.data_raw_RS.items() if k in source}
        
        # Things get easier if we can rely on having proper numbers as start and stop 
        if start is None:
            start =0
        if stop is None:
            stop = self.meta['length_total'] 
        
        # Now apply the start stop filter
        # We will handle the time separately
        for key in my_data.keys():
            my_data[key]=my_data[key][start:stop]
            
        return my_data
    
    @property
    def  sample_time(self):
        return self.meta['t_sample']
        
    @property
    def all_sources(self):
        return self.meta['source_names']
    
    @property
    def signal_sources(self):
        sources=self.meta['source_names']
        if 'Time' in sources:
            sources.remove('Time')
        return sources
    
    def getAsDf(self, start = None, stop = None, source=None, time=True):
        '''
        Returns a dataframe containin the data between start and stop. By default, also timestamps will be included.
        You can set time to False to exclue timestamps.
        This is useful for plotting or other processing of large files.
        
        :param start: Index of starting sample
        :param stop: Index of final sample
        :param source: Oscilloscope Source Name 
        :param time: Optional. Defaults to True. If false, no time data is returned.
        :type time: Boolean
        '''
        
        # Filtering is done a separate function. We get filtered data without time.
        my_data=self.getRaw(start = start, stop = stop, source=source)
        
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
                my_data[name]=self.data_raw_RS[name][start:stop]
            else:
                # Generate time stamps. We only generate them for the inteval between start and stop
                timestamps=self._genTimeAxis(start, stop)
                my_data['Time']=timestamps
        
        # Generate a DataFrame
        try:
            df = pd.DataFrame(my_data)
        except:
            print('Breakpoint')
            raise
        
        return df

    def _getLimitedStartStop(self):
        '''
        Returns an arbitrarily limited start and stop value you can apply to an RS_Det_File instance for fast testing. 
        '''
        MAXPLOTLEN = 5E7
        
        meta = self.meta        
        
        dfLen = meta['length_total']
        
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
    
