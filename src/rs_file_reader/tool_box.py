import sys
import os
from multiprocessing import Pool, cpu_count, Process
import json
import pickle
from copy import deepcopy

def _except_hook(cls, exception, traceback):
    '''
    Function used by :py:func:`pqt5_exception_workaround`. Don't call this directly.
    '''
    sys.__excepthook__(cls, exception, traceback)
    
class Cache():
    
    def __init__(self, file_base, enable_disk_cache):
        '''
        :param enable_disk_cache: If True on-disk cache will be used. Alternatively only. in-memory cache will be employed.
        '''
        self.enable_disk_cache=enable_disk_cache
        self.cache={}
        if enable_disk_cache:
            self.cache_file=file_base + '.cache'
            self.load()
            
    def write(self):
        '''
        Writes intermediate results to cache. 
        '''
        if self.enable_disk_cache: 
            with open(self.cache_file, 'wb') as cache:
                pickle.dump(self.cache, cache)

    def load(self):
        '''
        Load intermediate results from cache.
        '''
        if self.enable_disk_cache and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as cache:
                self.cache=pickle.load(cache)

    def get(self, fname, key):
        '''
        Returns a cached value for a given function name and key. Returns None if no cached value is available.
        '''
        if not fname in self.cache:
            self.cache[fname]={}
        
        if key in self.cache[fname]:
            return self.cache[fname][key]
        else:
            return None
            
    def set(self, fname, key, value):
        '''
        Sets a value for a given function name and key pair
        '''
        if not fname in self.cache:
            self.cache[fname]={}
    
def pqt5_exception_workaround():
    '''
    Workaround to prevent pyqt5 from silently eating exceptions.
    Call this at the beginning of your progame and you will experience relieve.
    '''
    sys.excepthook = _except_hook


class Persival():
    '''
    Keeps values persistent between program runs. 
    
    '''
    
    def __init__(self, persival_file='persival.json'):
        '''
        Will try to load parameters from the persival_file. 
        If the file does not exists this will be silently ignored and the file will be created when :py:meth:`save` will be called. 
    
        '''

        self.persival_file=persival_file        
        if os.path.exists(persival_file):
            with open(self.persival_file, 'r') as file:
                self.data=json.load(file)
        else:
            self.data={}
        
    def setDefault(self, name, value):
        '''
        Sets the given value only if no information was present in the persival file or if the persival file was not found.
        '''
        if not name in self.data.keys():
            self.data[name]=value
    
    def get(self, name):
        '''
        Returns the saved value for the given parameter.
        '''
        return self.data[name]
    
    def set(self, name, value):
        '''
        Store the given value in memory. You have to call save to dump the value to the disk. 
        '''
        self.data[name] = value
        
    def save(self):
        '''
        Saves the content to a file.
        '''
        with open(self.persival_file, 'w') as file: 
            json.dump(self.data, file)
    
class ProcessingHelper():
    '''
    Helper working around the limitations of Pool.
    
    '''
    def __init__(self, parallel=False):
        self.funcs=[]
        self.parallel=parallel
    
    def add(self, function, args, kwargs={}):
        '''
        Add a function to the processing pool.
        
        :param function: function to be executed
        :param args: list. Provice the positional arguments
        :param kwargs: dict. Optional, provide the kwargs
        '''
        self.funcs.append({'func': function, 'args': deepcopy(args), 'kwargs': deepcopy(kwargs)})
        
    def clear(self):
        self.funcs=[]
    
    def run(self):
        '''
        Start processing. 
        
        :returns: list of results. The orgdering matches the sequence of add calls.
        '''
        processes=cpu_count()
        # Windows does not handle full cpu loads well, so leave one CPU unutilized for Windows.
        if os.name == 'nt':
            processes -=1
            
        if self.parallel:
            with Pool(processes=processes) as pool:
                result=pool.map(self._myrun, range(len(self.funcs)))
        else:
            result=[]
            for i, f in enumerate(self.funcs):
                r=self._myrun(i)
                result.append(r)
                
        return result
    
    
    def __len__(self):
        
        return len(self.funcs)
    
    
    def _myrun(self, paramNo):
        '''
        wrapper around the function for pool
        
        :param paramNo:
        '''
        param=self.funcs[paramNo]
        func=param['func']
        args= param['args']
        kwargs= param['kwargs']
        retval=func(*args, **kwargs)
        return retval
        