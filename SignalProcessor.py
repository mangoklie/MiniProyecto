import subprocess
import wfdb
from os.path import exists

class SignalProcessor:

    START_WAVE = '('
    END_WAVE = ')'

    @staticmethod
    def process_wave(wave_seg):
        """
        Separates mixed annotations like '((pN))'. 
        This fucntion asumes that the first anotation found is the
        first to appear and so on. so '((pN))' is actually (p) (N).
        """
        symbols = ['p', "N", 't']
        wave_list = []
        aux_list = []
        while wave_seg:
            elem = wave_seg.pop(0)
            if elem[1] == SignalProcessor.START_WAVE:
                aux_list.append(elem)
                symbol_found = False
                i = 0
                for x in wave_seg:
                    if x[1] != SignalProcessor.START_WAVE and x[1] in symbols and not symbol_found:
                        if len(wave_seg) < 3:
                            aux_list.append(x)
                        else:
                            aux_list.append(wave_seg.pop(i))
                        symbol_found = True
                    elif x[1] == SignalProcessor.END_WAVE and symbol_found:
                        if len(wave_seg) < 3:
                            aux_list.append(x)
                        else:
                            aux_list.append(wave_seg.pop(i))
                        break
                    i += 1
                wave_list.append(aux_list)
                aux_list = []
        return wave_list


                

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
        self.record = wfdb.rsamp(sigfile)
    
    def detect_segments(self):
        """
        This method gets the a whole segment P-QRS-T form the annotantions 
        provided to be processed later
        """
        #Calculate RR segment (use ann2rr better and read the ouptu)
        symbols = ['p', "N", 't']
        annots = zip(self.annotations.sample,self.annotations.symbol,self.annotations.num)
        prev_n = []
        prevs = []
        aux_list = []
        open_count = 0
        for element in annots:

            if element[1] == SignalProcessor.START_WAVE:
                aux_list.append(element)
                open_count += 1
                continue
            elif  element[1] in symbols:
                aux_list.append(element)
                continue
            elif element[1] == SignalProcessor.END_WAVE:
                aux_list.append(element)
                open_count -=1
                if open_count:
                    continue
                segs = SignalProcessor.process_wave(aux_list)
                if len(segs) == 2:
                    #Calculate if a method is needed
                    pass
                elif segs == aux_list:
                    #This should be a function.
                    if prevs:
                        prev_symbol = prevs[1][1]
                        actual_symbol = aux_list[1][1]
                        if prev_symbol == 't' and actual_symbol == 'p':
                            #Calculate TP segment
                            pass
                        elif prev_symbol == 'p' and actual_symbol == 'N':
                            #Calculate PR segment, PR interval, left beat segment, left + mid 
                            pass
                        elif prev_symbol == 'N' and actual_symbol == 't':
                            #Calculate ST segment QT interval , left + right
                            pass
                
            else:
                raise ValueError('Symbol not recognized: ' + element[1])
