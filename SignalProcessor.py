import subprocess
import wfdb
import numpy as np
import scipy.stats as sps
import pywt
from os.path import exists
from functools import reduce
import argparse

class SignalProcessor:

    START_WAVE = '('
    END_WAVE = ')'
    CM_PER_SAMPLE =  2.5/300
    
    @staticmethod
    def detail_coefs_of_dwt(segment):
        _, detil_coefs = pywt.dwt(segment,'db1')
        return detil_coefs

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
        self.sigfile = sigfile

    def __processSegments(self,prevs,aux_list, prev_n):
        prev_symbol = prevs[1][1]
        actual_symbol = aux_list[1][1]
        if  actual_symbol == 'p':
            #Calculate TP segment
            if prev_symbol == 't':
                self.segments['tp_segment'].append(SignalProcessor.calc_tp_segment(prevs, aux_list))
            self.segments['p_wave'].append((aux_list[0][0], aux_list[1][0] , aux_list[-1][0]))

        elif actual_symbol == 'N':
            #Calculate PR segment, PR interval, left beat segment, left + mid 
            if prev_symbol == 'p':
                self.segments['pr_segment'].append(SignalProcessor.calc_pr_segment(prevs, aux_list))
                self.segments['pr_interval'].append(SignalProcessor.calc_pr_interval(prevs, aux_list))
                self.segments['left_mid'].append(SignalProcessor.calc_left_mid(prevs, aux_list))
            self.segments['qrs_complex'].append((aux_list[0][0], aux_list[1][0], aux_list[-1][0]))
            
        elif actual_symbol == 't':
            #Calculate ST segment QT interval (mid + right)
            self.segments['t_wave'].append((aux_list[0][0], aux_list[1][0], aux_list[-1][0]))
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

    def calculate_heart_rate(self):
        """ Calls the WFDB program hrstats to get the information about the 
            ECG heart rate. 

            returns: mean heart rate and deviation
        """
        completedProcess = subprocess.run('hrstats', '-r', self.sigfile, '-a', 'annot', stdout = subprocess.PIPE)
        result = completedProcess.stdout
        result = result.split()
        bpm = result[1].split('|')
        bpm = bpm[1].split('/')
        bpm = bpm[1]
        return bpm, abs(eval(result[2])) #beats per minute and desviation

    def get_mean_rr_value(self, save_rr_metric = False):
        """
            Calls the WFDB program ann2rr to get a list of the RR intervals in samples and then gets the mean
            returns the mean of the RR intervals in seconds
        """
        completedProcess = subprocess.run(['ann2rr','-r',self.sigfile,'-a','annot'], stdout = subprocess.PIPE)
        rr_segments_length = completedProcess.stdout
        rr_segments_length = rr_segments_length.split()
        aux_val = list(map(eval,rr_segments_length))
        if save_rr_metric:
            self.segments['rr_interval'] = aux_val
        mean_rr_ = np.mean(aux_val) / self.record.fs

    def get_interval_wave_durations(self, name):
        """
            General function to get the interval or wave durations 
        """
        segment = self.segments.get(name)[:]
        return np.array(list(map(lambda x: x[-1]-x[0], segment))) / self.record.fs
    
    def get_pr_intervals(self):
        """
            Calculates PR interval values from the pr_interval markers.
            Returns a list with the intervals in seconds
        """
        pr_intervals = self.segments.get('pr_interval')[:]
        pr_intervals = np.array(list(map(lambda x: x[-1] - x[0], pr_intervals))) / self.record.fs
        return pr_intervals

    def get_p_wave_durations(self):
        """
            Transforms the  p_wave segment marks and returns the p_wave duration in seconds. 
        """
        p_waves = self.segments.get('p_wave')[:]
        return np.array(list(map(lambda x: x[-1]-x[0], p_waves))) / self.record.fs

    def get_qt_interval_durations(self):
        """
            Transforms the qt interval segment marks and returns the qt_itnerval duration in seconds
        """
        qt_intervals = self.segments.get('qt_interval')[:]
        return np.array(list(map(lambda x: x[-1]- x[0], qt_intervals))) / self.record.fs

    def get_t_wave_durations(self):
        t_wave_segments = self.segments.get('t_wave')[:]
        return np.array(list(map(lambda x: x[-1] - x[0],t_wave_segments))) / self.record.fs
    
    #Mean entropy for PR interval durations
    def get_pr_interval_durations_entropy(self):
        pr_intervals = self.get_pr_intervals()
        hist, bin_edges = np.histogram(pr_intervals,'auto')
        bin_map_pr_interval = np.digitize(pr_intervals,bin_edges[:-1])
        #probabilities_list = [ hist[i]/len(pr_intervals) for i in range(len(hist))]
        #bin_map_pr_interval = map(lambda x: probabilities_list[x])
        bin_map_pr_interval = np.array(list(map(lambda x: hist[x]/len(pr_intervals), bin_map_pr_interval)))
        return sps.entropy(bin_map_pr_interval)
    
    #Mean entropy por P wave durations
    def get_p_wave_durations_entropy(self):
        p_wave_durations = self.get_p_wave_durations()
        hist, bin_edges = np.histogram(pr_intervals, 'auto')
        bin_map_p_waves = np.digitize(p_wave_durations, bin_edges[:-1])
        bin_map_p_waves = np.array(list(map(lambda x: hist[x]/len(p_wave_durations), bin_map_p_waves)))
        return sps.entropy(bin_map_p_waves)

    #Mean entropy for QT intervals
    def get_qt_interval_entropy(self):
        p_wave_durations = self.get_qt_interval_durations()
        hist, bin_edges = np.histogram(pr_intervals, 'auto')
        bin_map_p_waves = np.digitize(p_wave_durations, bin_edges[:-1])
        bin_map_p_waves = np.array(list(map(lambda x: hist[x]/len(p_wave_durations), bin_map_p_waves)))
        return sps.entropy(bin_map_p_waves)
    
    #Mean entropy of rr interval durations
    def get_rr_interval_durations_entropy(self):
        p_wave_durations = self.segments.get('rr_interval')
        hist, bin_edges = np.histogram(pr_intervals, 'auto')
        bin_map_p_waves = np.digitize(p_wave_durations, bin_edges[:-1])
        bin_map_p_waves = np.array(list(map(lambda x: hist[x]/len(p_wave_durations), bin_map_p_waves)))
        return sps.entropy(bin_map_p_waves)

    #Mean Entropy of T wave durations
    def get_t_wave_durations_entropy(self):
        t_wave_durations = self.get_t_wave_durations()
        hits, bin_edges = np.histogram(t_wave_durations, 'auto')
        bin_map_t_waves = np.digitize(t_wave_durations, bin_edges[:-1])
        bin_map_t_waves = np.array(list(map(lambda x: hist[x]/len(t_wave_durations), bin_map_t_waves)))
        return sps.entropy(bin_map_t_waves)

    # Area under highest frequency of RR durations (?)
    # Area under lowest frequency of RR durations (?)
    # Beats per minute (See calculate_heart_rate)
    
    # Is the P wave inverted?
    # Is the QRS complex inverted?
    # Is the T wave inverted?
    def wave_inverted(self, name = 'p_wave'):
        p_waves = self.segments.get(name)[:]
        p_waves = list(map(lambda x: self.record.p_signals[x[1]] <= self.record.p_signals[x[0]] and self.record.p_signals[x[1]] <= self.record.p_signals[x[-1]],
        p_waves))
        return np.sum(p_waves) / len(p_waves) > .75
    
    # Mean duration of PR, QT, RR intervals, P and T waves and QRS complexes
    def mean_duration_of_interval(self, interval_name):
        """
            Returns the mean duration of a given inerval 
        """
        interval = self.segments.get(interval_name)[:]
        interval = list(map(lambda x: x[-1]-x[0], interval))
        return np.mean(interval) / self.record.fs

    def mean_amplitude_of_wave(self, wave):
        """
        Returns the mean amplitude of a given wave in cm.
        """
        assert wave in ['p_wave', 't_wave', 'qrs_complex'], "Only QRS complex or P or T waves"
        aux_rec = self.record.p_signals
        wave = self.segments.get(wave)
        apmplitudes = list(map(lambda x: aux_rec[x[1]] - (aux_rec[x[0]] + aux_rec[x[-1]])/2 , wave))
        return np.mean(apmplitudes)

    def rr_difs(self):
        """
        Returns the consecutive differences between the RR intervals in seconds
        """
        rrs = np.array(self.segments.get('rr_interval'))
        delta_rrs = (rrs[1:]-rrs[:-1])/self.record.fs
        return delta_rrs

    # Proportion of consecutive differences of RR greater than 20ms or than 50ms
    def rr_difs_prop_greather_than(self, threshhold = 0.02):
        """
        Calculates the proportion of consecutive differences of RR intervals greater than a
        given threshold. 
        """
        delta = self.rr_difs()
        return np.sum(delta > threshold)/len(delta)

    # Root mean square of consecutive differences of RR interval durations
    def root_mean_square_of_rr_differences(self):
        delta = self.rr_difs()
        return np.sqrt(np.mean(delta**2))
    
    #Standard deviation or P, T wave duration, QRS complex, PR interval, QT interval, consecutive differences of RR interval durations
    # RR interval durations, P, R, T peak amplitudes
    def sd_of_durations(self, name):
        segments = self.get_interval_wave_durations(name)
        return numpy.std(segments)

    def sd_of_amplitudes(self,wave):
        """
        Returns the standard deviation of the amplitudes of the P, T or QRS waves. 
        """
        assert wave in ['p_wave', 't_wave', 'qrs_complex'], "Only QRS complex or P or T waves"
        aux_rec = self.record.p_signals
        wave = self.segments.get(wave)
        apmplitudes = list(map(lambda x: aux_rec[x[1]] - (aux_rec[x[0]] + aux_rec[x[-1]])/2 , wave))
        return np.std(apmplitudes)

    def sd_of_rr_difs(self): # Maybe delete this one 
        return np.std(self.rr_difs())

    #Mean amplitude on left, right and mid segments are (I think the mean of al samples.)
    #Use pywavelets for the wavelet

    def get_segment(self,segment):
        segments = self.segments.get(segment)[:]
        return map(lambda x: self.record.p_signals[x[0]: x[-1] +1], segments)

    def mean_amplitude_on_segments(self,segment):
        segments = reduce(lambda x,y: np.concatenate((x,y)), self.get_segment(segment))
        return np.mean(segment)

    def variance_amplitude_segments(self,segment):
        segments = reduce(lambda x,y: np.concatenate((x,y)), self.get_segment(segment))
        return np.var(segments)

    def skewnes_segment(self, segment):
        segments = reduce(lambda x,y: np.concatenate((x,y)), self.get_segment(segment))
        return sps.skew(segments)

    def kurtosis_of_segment(self, segment):
        segments = reduce(lambda x,y: np.concatenate((x,y)), self.get_segment(segment))
        return sps.kurtosis(self.get_segment(segment))

    def wavelet_detail_coefs(self, segment):
        segments = map(SignalProcessor.detail_coefs_of_dwt,self.get_segment(segment))
        return segments

    def mean_wavelet_detail_coefs(self,segment):
        segments = reduce(lambda x,y: np.concatenate((x,y)), self.wavelet_detail_coefs(segment))
        return np.mean(segments)

    def mean_kurtosis_wavelet_detail_coefs(self,segment):
        segments = map(sps.kurtosis, self.wavelet_detail_coefs(segment))
        return np.mean(list(segments))

    def mean_skew_wavelet_detail_coefs(self,segment):
        segments = map(sps.skew, self.wavelet_detail_coefs(segment))
        return np.mean(list(segments))

    def mean_std_wavelet_detal_coefs(self,segment):
        segments = map(np.std, self.wavelet_detail_coefs(segment))
        return np.mean(list(segments))


















        

        



