

from abc import ABC

class Plotting(ABC):
    
    def __init__(self, results):
        self._results = results
        self._data_cont = []
        self._stim_amp = []
        self._timebursts = []
        self.
        
        
    def _unpack_results(self):
        for data in self._results:
            self._data_cont = data_run['data']['erna_cont']
        
        
    @property
    def results(self):
        return self._results



def plot_single_run_results():
    pass


def plot_group_level_results():
    pass
