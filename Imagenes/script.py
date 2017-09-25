import skimage.io as sio
import skimage.filters as sfil
import skimage.exposure as sexp
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os
from skimage.transform import hough_line, houg_line_peaks
from scipy.interpolate import interp1d
from scipy.misc import imsave
from skimage.viewer import ImageViewer

plt.switch_backend('qt5agg')

class SignalRetriever():

    fun = lambda x, y, z :np.logical_or(x, np.logical_or(y, z))
    view_func = lambda x: ImageViewer(x).show()

    @staticmethod
    def get_baseline(image_src):
        pass

    @staticmethod
    def transform_to_binary_image(image_src, corrector=35):
        #La funcion transforma la imagen a blanco y negro haciendo su mejor esfuerzo 
        #por preservar las líneas de la señal del ECG.
        
        # Esta funcion realiza resalta los negros dado un determinado factor para 
        # visualizar la senial.
        fun_corrector = lambda x : lambda y: int(x==y)*y*corrector//100 + int(x!=y)*y
        fun = SignalRetriever.fun

        #Lectura de la imagen y posterior ajuste de Gamma (Para que eliminar difuminadso)
        image = sexp.adjust_gamma(image_src,1.25)
        
        #Datos importantes para la imagen 
        try:
            image_aux = np.reshape(image,(image.shape[0]*image.shape[1],image.shape[2]))
        except IndexError:
            pass

        #El color que mas aparece
        color_dominance= np.sum(image_aux,0)
        color_max = np.max(color_dominance)
        index_max = np.where(color_dominance==color_max)
        index_max = index_max[0][0]

        #Threshold para cada canal de la imagen
        rgb_t = list(map(lambda x: sfil.threshold_isodata(x),[image_aux[:,i] for i in range(3) ]))
        fun_corrector = fun_corrector(rgb_t[index_max])

        #Aplicando threshold a cada canal y unificando
        image_aux = fun(image_aux[:,0]>fun_corrector(rgb_t[0]),
            image_aux[:,1]>fun_corrector(rgb_t[1]),
            image_aux[:,2]>fun_corrector(rgb_t[2]))
        
        #Redimensionando la imagen.
        bn_im =  np.reshape(image_aux,(image.shape[0],image.shape[1]))
        return bn_im
        

    @staticmethod
    def retrieve_signal(points, image_src):
        # Esta funcion toma los pixeles marcados y realiza una interpolacion con los pixeles 
        # obtenido del procesamiento de la imagen
        y_standarized = (image_src.shape[0]-points[0]).astype(float)
        inter_func = interp1d(points[1],y_standarized)
        return inter_func

    @staticmethod
    def plot_digital_ecg(inter_func,points):
        #Funcion de prueba para ver como se ve la señal luego de una interpolación.
        x = np.linspace(np.min(points[1]),np.max(points[1]),750)
        plt.plot(x,inter_func(x))
        plt.axes().set_aspect('equal','box')
        plt.show()

    @staticmethod
    def sample_signal(inter_func,points):
        # Guardando archivo de la señal. 
        x = np.linspace(np.min(points[1]),np.max(points[1]),750)
        x = inter_func(x)
        sdx = np.std(x)
        x-=np.mean(x)
        x/=sdx
        x = np.reshape(x,(x.shape[0],1))
        # record = wfdb.Record(recordname='Test1',fs=300,nsig=1,siglen=750,p_signals=x,
        # filename=['test.dat'],baseline=[-1],units=['mV'],signame=['ECG'])
        # wfdb.plotrec(record,title='Test')
        return x


    def __init__(self,*args,**kwargs):
        self.images_src = args
        self.dir = kwargs.get('dir')
        exclude = kwargs.get('exclude')
        # Hay archivos para excluir ?
        if exclude:
            #exclude_files = open(exclude,'r')
            exclude_files = ['img_0.jpg','img_0_II_long.jpg']
            # json parse
            self.images_src = list(map(lambda x: self.dir+'/'+x, filter(lambda x: x not in exclude_files or x[-6:] =='bn.jpg',self.images_src)))
    
    def get_multisignal_from_images(self):
        images = sio.imread_collection(self.images_src)
        i = 0
        array_signal = []
        for image in images:
            points, bn_image = SignalRetriever.transform_to_binary_image(image,corrector=15)
            #imsave(self.dir+'/'+'img_{0}_bn.jpg'.format(i),bn_image)
            inter_func = SignalRetriever.retrieve_signal(points,image)
            x = SignalRetriever.sample_signal(inter_func,points)
            # record = wfdb.Record(recordname='Test'+str(i),fs=300,nsig=1,siglen=750,p_signals=x,
            # filename=['test.dat'],baseline=[50],units=['mV'],signame=['ECG'],adcgain=[200],fmt=['16'],checksum=[0],
            # adcres=[16],adczero=[0],initvalue=[0],blocksize=[0], d_signals=None)
            #array_signal = np.concatenate((array_signal,SignalRetriever.sample_signal(inter_func,points)))
            array_signal.append(x)
            i+=1
        return array_signal

if __name__ == '__main__':
    
    sr = SignalRetriever(*os.listdir('img0'),exclude=True,dir='img0')
    array_signal = sr.get_multisignal_from_images()
    # m_record = wfdb.MultiRecord(recordname='Test', segments=array_signal,nsig=len(array_signal),
    # siglen=9000,
    # fs=300,
    # segname=[str(i) for i in range(len(array_signal))],
    # seglen=[750]*len(array_signal))

    wfdb.wrsamp(recordname='Test',fs=300, units=['mV'],signames=['Lead_X'],
    p_signals = array_signal[0],fmt=['16'],baseline=[-1],gain=[1000])

    

