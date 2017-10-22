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
    
    @staticmethod
    def calc_tp_segment(t_seg, p_seg):
        return (t_seg[-1][0],p_seg[0][0])

    @staticmethod
    def calc_pr_segment(p_seg, qrs_complex):
        return (p_seg[-1][0],qrs_complex[0][0])

    @staticmethod
    def calc_pr_interval(p_seg, qrs_complex): #Letf beat (?)
        return (p_seg[0][0], qrs_complex[0][0])

    @staticmethod
    def calc_left_mid(p_seg, qrs_complex):
        return (p_seg[0][0],qrs_complex[-1][0])

    @staticmethod
    def calc_st_segment(qrs_complex, t_seg):
        return (qrs_complex[-1][0], t_seg[0][0])

    @staticmethod
    def calc_qt_interval(qrs_complex, t_seg): # mid + right
        return (qrs_complex[0][0], t_seg[-1][0])         

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
        self.record = wfdb.rdsamp(sigfile)
        self.segments = {
            'p_wave': [],
            't_wave': [],
            'qrs_complex': [],
            'pr_segment': [],
            'pr_interval': [],
            'left_mid': [],
            'tp_segment': [],
            'qt_interval': [],
            'st_segment': []

        }

    def __processSegments(self,prevs,aux_list, prev_n):
        prev_symbol = prevs[1][1]
        actual_symbol = aux_list[1][1]
        if  actual_symbol == 'p':
            #Calculate TP segment
            if prev_symbol == 't':
                self.segments['tp_segment'].append(SignalProcessor.calc_tp_segment(prevs, aux_list))
            self.segments['p_wave'].append((aux_list[0][0], aux_list[-1][0]))

        elif actual_symbol == 'N':
            #Calculate PR segment, PR interval, left beat segment, left + mid 
            if prev_symbol == 'p':
                self.segments['pr_segment'].append(SignalProcessor.calc_pr_segment(prevs, aux_list))
                self.segments['pr_interval'].append(SignalProcessor.calc_pr_interval(prevs, aux_list))
                self.segments['left_mid'].append(SignalProcessor.calc_left_mid(prevs, aux_list))
            self.segments['qrs_complex'].append((aux_list[0][0], aux_list[-1][0]))
            
        elif actual_symbol == 't':
            #Calculate ST segment QT interval (mid + right)
            self.segments['t_wave'].append((aux_list[0][0], aux_list[-1][0]))
            if prev_symbol == 'N':
                self.segments['qt_interval'].append(SignalProcessor.calc_qt_interval(prevs, aux_list))
                self.segments['st_segment'].append(SignalProcessor.calc_st_segment(prevs, aux_list))
    
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
                if not open_count:
                    continue
                aux_list.append(element)
                continue
            elif element[1] == SignalProcessor.END_WAVE:
                if open_count -1 < 0 and not open_count:
                    continue
                aux_list.append(element)
                open_count -=1
                if open_count and open_count > 0:
                    continue
                segs = SignalProcessor.process_wave(aux_list[:])
                if len(segs) >1:
                    #Calculate if a method is needed
                    print('here',segs)
                    for seg  in segs:
                        if prevs:
                            self.__processSegments(prevs,seg,prev_n)
                            if seg[1][1] == "N":
                                prev_n = seg
                        prevs = seg
                elif segs[0] == aux_list: #ActiveBNK pass 0815 
                    print('bellow here')
                    if prevs:
                        self.__processSegments(prevs,aux_list, prev_n)
                
                if aux_list[1][1] == 'N':
                    prev_n = aux_list
                prevs = aux_list
                aux_list = []
            else:
                raise ValueError('Symbol not recognized: ' + element[1])
