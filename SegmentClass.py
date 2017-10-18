
class Segment:

    def __init__(self,*args,**kwargs):
        self.p_wave_start = kwargs.get('p_wave_start')
        self.p_wave_peak = kwargs.get('p_wave_peak')
        self.p_wave_end = kwargs.get('p_wave_end')
        self.qrs_start = kwargs.get('qrs_start')
        self.qrs_peak = kwargs.get('qrs_peak')
        self.qrs_end = kwargs.get('qrs_end')
        self.t_wave_start = kwargs.get('t_wave_start')
        self.t_wave_peak = kwargs.get('t_wave_peak')
        self.t_wave_end = kwargs.get('t_wave_end')

    def calculate_left_beat(self):
        pass

    def calculate_qrs_segment(self):
        pass

    def calculate_right_beat(self):
        pass

    def calculate_qt_interval(self):
        pass

    def calculate_pr_segment(self):
        pass

    def calculate_left_plus_qrs(self):
        pass

    def calculate_qrs_plus_right(self):
        pass

    #RR and TP segments are inter beat segment 