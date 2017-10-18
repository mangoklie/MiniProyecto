import subprocess
import wfdb
from os.path import exists

class SignalProcessor:

    def __init__(self,*args,**kwargs):
        if not args:
            raise ValueError("No signal file was specified")
        #The first args value must be the signal filepath
        sigfile = args[0]
        if exists(sigfile + '.annot'):
            self.annotations = wfdb.rdann(recordname = sigfile, extension = 'annot')
        else:
            process_summary = subprocess.run(['ecgpuwave', '-r', sigfile, '-a', 'annot'])
            if process_summary.returncode:
                raise ValueError("Expected process to resturn code 0")
            self.annotations = wfdb.rdann(recordname = sigfile, extension = 'annot')
    
    def detect_whole_segments(self):
        """
        This method gets the a whole segment P-QRS-T form the annotantions 
        provided to be processed later
        """
        pass
    
    

