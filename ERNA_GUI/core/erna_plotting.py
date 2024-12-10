

from abc import ABC

class Plotting(ABC):
    
    def __init__(self, results):
        self._prepare_results(results)
        
        
        
    def _prepare_results(self,results): 
        self._run = results['run']
        self._task = results['task']
        self._rrun_details= results['run_details']
        self._stimChannels = results['stimChannels']
        self._time_gtc = results['time_gtc']
        self._run = results['run']
        self._run = results['run']
        self._run = results['run']
        self._run = results['run']
        
        data_run['run'].append(run)
        data_run['task'].append(task)
        data_run['run_details'].append(result.get('run_details', None))
        data_run['stimChannels'].append(result.get('stimChannels', None))
        data_run['time_gtc'].append(result['data'].get('time', None))
        data_run['data_cont'].append(result['data'].get('erna_cont', None))
        data_run['data_erna'].append(result['data'].get('erna_data', None))
        data_run['timebursts'].append(result['data'].get('timebursts', None))
        data_run['stim_amp'].append(result['data'].get('stim_amp', None))
            
        
    @property
    def results(self):
        return self._results



def plot_single_run_results():
    
    pass


def plot_group_level_results():
    pass
