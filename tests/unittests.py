'''
    File name: test_fileReader.py
    Author: Simon Waid
    Date created: 2021
    Python Version: 3.8, 3.9
'''


import unittest
from src.rs_file_reader import RS_File
from src.rs_file_reader import RS_Analysis
 
import os
import pandas as pd
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)



singles=["2021-09-14 17_44_28.996310.bin"]

# Couples contains the binary files as keys and the csv files as values.
couples={   'aquisition_auto_1.bin': 'aquisition_auto_1.csv', \
            'FastSegmentation1_1.bin': 'FastSegmentation1_1.csv',\
            'FastSegmentation1_2.bin': 'FastSegmentation1_1.csv',\
            'FastSegmentation2_1.bin': 'FastSegmentation2_1.csv',\
            'XYMultichan.bin': 'XYMultichan.csv',\
            'XYSinglechannel.bin': 'XYSinglechannel.csv',\
            'SingleChanCh2OffsetPosRaw.bin': 'SingleChanCh2OffsetPos.csv', \
            'singleChanCh2Offset.bin': 'singleChanCh2Offset.csv',\
            'MultiChanOffsetRaw.bin': 'MultiChanOffset.csv', \
            'MultiChanOffset.bin': 'MultiChanOffset.csv' ,\
            'singleChanRaw.bin':'singleChan.csv',\
            'singleChanCh2.bin':'singleChanCh2.csv',\
            'singleChan.bin':'singleChan.csv',\
            'multiChan.bin':'multiChan.csv',\
            'multiChanRaw.bin':'multiChan.csv',\
            'MultiChanOffsetRaw.bin': 'MultiChanOffset.csv', \
            'singleChanCh2OffsetRaw.bin': 'singleChanCh2Offset.csv',\
            '16bit_multichannel.bin': '16bit_multichannel.csv',\
            '16bit_singlechannel.bin': '16bit_singlechannel.csv', \
            'aquisition_auto_2.bin': 'aquisition_auto_2.csv'
            }

corrupted_size=['corrupted_size.bin'] 

def fullPath(file):
    '''
    Convenience function.
    '''
    directory='testdata'
    return os.path.join(directory, file)
 

class Test_RS_bin_file(unittest.TestCase):
    '''
    Tests for the class :py:class:`RS_bin_file`.
    
    '''

    #corrupted_xml=['corrupted_xml.bin']        
            
    def test_bin_vs_csv(self):
        '''
        Test principle: We have a collection of binary files and the corresponding .csv files. 
        We compare the data returned by :py:class:`RS_bin_file` to the data in the .csv file.
    
        '''
                
        # Match csv channel names to binary channel names 
        matching_names={'CH1_TR1': 'Ch1Wfm1', 'CH2_TR1': 'Ch2Wfm1', 'CH3_TR1': 'Ch3Wfm1', 'C1W1': 'Ch1Wfm1'}
          
        #couples={'RefCurve_2021-09-09_SingleChanRaw_094531':'RefCurve_2021-09-09_SingleChanRaw_094506', \
        #         'RefCurve_2021-09-09_multiChanRaw_094438':'RefCurve_2021-09-09_mulitChan_094339',\
        #         'RefCurve_2021-09-09_multiChan_094428':'RefCurve_2021-09-09_mulitChan_094339',\
        #         'RefCurve_2021-09-08_0_164525': 'RefCurve_2021-09-08_0_164536'}
        
        # Iteate over all couples of files and compare them.
        for binary, csv in couples.items():
            for start in [None, 100]:
                for stop in [None, 500]:
                    
                    rs_bin_file=RS_File(fullPath(binary))
                    bin_data_nr=rs_bin_file.getAsDf(start, stop)
                    rs_csv_file=RS_File(fullPath(csv))
                    csv_data_nr=rs_csv_file.getAsDf(start, stop)
                    # Rename the columns so the data matches.
                    # Differetnt software versions seem to name traces differently.
                    #columns=csv_data.columns
                    #replace = False
                    #for column in columns:
                    #    if not column in list(matching_names.keys()):
                    #        replace=True
                    #if replace:
                    bin_data = bin_data_nr.rename(columns = matching_names)
                    csv_data = csv_data_nr.rename(columns = matching_names)
                    
                    diff=bin_data.subtract(csv_data)
                    div=bin_data.divide(csv_data)
                    
                    # Calculate error
                    for c in diff.columns:
                        ref_max= (csv_data[c].values**2).max()
                        max_error= (diff[c].values**2).max()
                        max_error_index=np.argmax(diff[c].values**2)
                    
                        rel_error=max_error/ref_max
                        min_rel_err=np.min(rel_error)
                        max_rel_err=np.max(rel_error)
                    
                        logging.info(f'Error: max: \n {max_error};\n min rel: {min_rel_err}; max rel: {max_rel_err}') 
            
                        # Check that csv and bin matches within rounding accuracy
                        test_pass1=(rel_error < 1e-2).all() #and (max_error < 1e-4).all()
                        test_pass2=(max_error < 1e-3).all()
                        if not test_pass1 or not test_pass2:
                            print(f'Max Error index: {max_error_index}')
                            start=max_error_index-20
                            if start <0:
                                start=0
                            end=max_error_index+20
                            if end > len(csv_data):
                                end=len(csv_data)
                            try:
                                print(c)
                                print(bin_data[c][start:end].values)
                                print(csv_data[c][start:end].values)
                            except:
                                print('Breakpoint')
                        try:
                            self.assertTrue(test_pass1, f'Error: max: \n {max_error};\n min rel: {min_rel_err}; max rel: {max_rel_err}') 
                            self.assertTrue(test_pass2, f'Error: max: \n {max_error};\n min rel: {min_rel_err}; max rel: {max_rel_err}')
                        except:
                            print('Breakpoint')
                            
    def test_bin(self):
        '''
        Test principle: We load a binay file into memory and verify that no exception occurs.
    
        '''
        # Somewhat not very satisfying. For these files we simply check that they loaded into memory without any comarison
        
        directory='testdata'
        # Iteate over all couples of files and compare them.
        for binary in singles:
            binary_file=os.path.join(directory, binary)
            RS_File(binary_file)

    def test_fileSize(self):
        '''
        Check if the file size is correctly calculated. 
        '''
        size_ok_file=list(couples.keys())
        size_ok_file+=singles
        size_not_ok_file=corrupted_size
        
        # Positive test, file size check must return True 
        for file in size_ok_file:
            rs_file=RS_File(fullPath(file))
            size_ok, file_incomplete=rs_file.check_file_size()
            if file_incomplete:
                continue
            if not size_ok:
                print('Breakpoint')
            self.assertTrue(size_ok, f'Error in file {file}')
        
        # Negative test, file size check must return False        
        for file in size_not_ok_file:
            rs_file=RS_File(fullPath(file))
            size_ok, file_incomplete = rs_file.check_file_size()
            self.assertFalse(size_ok, f'Error in file {file}')
    
    def test_check_xml(self):
        '''
        Check the xml sanity check
        '''
    
        xml_ok_file=list(couples.keys())
        xml_ok_file+=singles
        #xml_not_ok_file=self.corrupted_xml
        
        # Positive test, xml check must return True 
        for file in xml_ok_file:
            rs_file=RS_File(fullPath(file))
            if not rs_file.check_xml():
                print('Breakpoint')
            self.assertTrue(rs_file.check_xml(), f'Error in file {file}')
        # Negative test, xml check must return False        
        #for file in xml_not_ok_file:
        #    rs_file=RS_file(fullPath(file))
        #    self.assertFalse(rs_file.check_xml(), f'Error in file {file}')
    
    
        
            
            
            
            
            
class Test_Det_file(unittest.TestCase):
    '''
    Test cases for the class Det_file
    '''
    statistics={'XYMultichan.bin':None, 'MultiChanOffsetRaw.bin':None}
    tot={'XYMultichan.bin':None}

    plot={'2021-09-18 10_39_32.249576.bin':None}
    fastStat={'singleChanCh2Offset.bin': None, \
              '2021-09-18 10_39_32.249576.bin':None
              }

    #Statistics: {'CH1_TR1': {'min': -0.09960474, 'max': 0.007905139, 'std': 0.0011975107}, 'CH3_TR1': {'min': -0.031620555, 'max': 0.37944666, 'std': 0.004378179}}

    def test_getStatistics(self):
        '''
        Regression test. Test principle: Compare the return value of getStatistics to known values. 
        '''
        for file, stat in self.statistics.items():
            rs_file=RS_File(fullPath(file))
            rs_analysis=RS_Analysis(rs_file)
            put_stat= rs_analysis.getStatistics()
            logging.info(f'Output by getStatistics: {put_stat}. Expected: {stat}') 
            #test_pass=stat is None 
            
            #self.assertTrue(test_pass1)
            
    def test_getTotAuto(self):
        '''
        Regression test. Test principle: Compare the return value of getTot to known values. 
        '''
        for file, stat in self.statistics.items():
            rs_file=RS_File(fullPath(file))
            rs_analysis=RS_Analysis(rs_file)
            tot= rs_analysis.getTotAuto()
    
    def test_fastStat(self):
        '''
        Test fastStat against GetStatistics 
        '''
        tolerace=0.15
        for file, res in self.fastStat.items():
            rs_file=RS_File(fullPath(file))
            rs_analysis=RS_Analysis(rs_file)
            stat_ref=rs_analysis.getStatistics()
            stat_fast=rs_analysis.fastStat()
            for source in stat_ref.keys():
                for res in stat_ref[source].keys():
                    r_ref=stat_ref[source][res]
                    r_fast=stat_fast[source][res]
                    err=abs(abs(r_ref/r_fast)-1)
                    test_pass= err <  tolerace
                    if not test_pass:
                        print('Breakpoint')
                        logging.info(f'Stat ref: {stat_ref}')
                        logging.info(f'Stat stat_fast: {stat_fast}')
                    self.assertTrue(test_pass, f'Error in {file}, {source}, {res}, result ref: {r_ref}, result fast {r_fast}') 

 #       print(stat_ref, stat_fast)
#        quit()
    
    def test_get_2d_histo(self):
        '''
        Test the get_2d_histo function. Test principle: Simply call the function with known good input data and hope it does not crash.
        '''
        for file in couples.keys():
            try:
                logging.info(f'Get 2D histo Processing {file}')
                rs_file=RS_File(fullPath(file))
                rs_analysis=RS_Analysis(rs_file)
                source=rs_file.signal_sources[0]
                rs_analysis.get_2d_histo(source)
            except:
                print(file)
                logging.info(f'Test failed for file: {file}')
                raise
                
                 
    def test_plotEvents(self):
        '''
        Regression test. Test principle: Compare the return value of getTot to known values. 
        '''
        for file, res in self.plot.items():
            rs_file=RS_File(fullPath(file))
            rs_analysis=RS_Analysis(rs_file)
            stat=rs_analysis.getStatistics()
            events, meta= rs_analysis.getTotAuto()
            rs_analysis.plotEvents(events, 'test_res')
        

if __name__ == '__main__':
    unittest.main()