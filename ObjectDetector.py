
import matplotlib.pyplot as plt
import numpy as np
import re
from os.path import isdir
from os import listdir
from sklearn.cluster import AgglomerativeClustering, KMeans, dbscan
from sklearn.neighbors import kneighbors_graph
from skimage import restoration
from skimage.draw import polygon, polygon_perimeter, circle
from skimage.color import rgb2gray, label2rgb
from skimage.io import imread, imsave, imread_collection, imshow
from skimage.filters import threshold_otsu
from skimage.morphology import square, binary_closing, closing, dilation
from skimage.transform import resize
from skimage import measure
from operator import mul
from functools import reduce
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file_path', help = "path to the file or directory to be oppened")
parser.add_argument('-m', '--matrix', help = "Path to the matrix output file ")
parser.add_argument('-i', '--imagepath', help= "Save the object detected image to a path")
parser.add_argument('-o', "--outdir", help = "Directory where to ouput the processed matrix images")
parser.add_argument('-imd', '--imagedir', help = "Directory where to put the labeled images" )
parser.add_argument('-p', '--patch', help = "Patch regions considered to be noise")
parser.add_argument('-pd', '--patchdir', help = "Location of patched images")
parser.add_argument('-c','--cluster', help = "Once patched perform clustering")
parser.add_argument('-cd', '--clusterdir', help = "Cluster dir for batch clurstering images")


class ObjectDetector(object):
    """
    Class to identify shapes in the B/W ECG IMAGE
    """

    RECTANGLE_TOLERANCE = 0.80


    
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            raise TypeError('No arguments provided')
        elif len(args) == 1:
            self.img_src = imread(args[0])
            self.regions = []

    
    def get_image_area(self):
        return reduce(mul,self.img_src.shape)
    
    def object_detect(self):
        """
        This method returns a labeled image with acording to the similarity of the pixels in a
        certain area. Similar pixels are considered to belong to the same object
        """
        image = self.img_src.copy()
        image = restoration.denoise_tv_chambolle(image, weight=0.1)
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(2))
        cleared = bw.copy()
        label_image = measure.label(cleared, connectivity=2)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1
        return label_image

    def __get_objects_coordinates(self,labeled_image = None):
        """
        Gets the coordinates of the square surrounding the object and rest
        """
        if self.regions:
            return self.regions
        l_image = labeled_image            
        regions = []
        for region in measure.regionprops(l_image):
            if region.area > self.get_image_area() * 0.005:
                continue
            minr, minc, maxr, maxc = region.bbox
            regions.append((minr, maxr, minc, maxc))
        self.regions = regions
        return self.regions

    def crop_resize_ravel(self,region, shape = (20,20), ravel = True):
        """
        crops resizes and ravels the image 
        """
        minr, maxr, minc, maxc = region
        img_section = self.img_src[minr:maxr, minc:maxc]
        sample = resize(img_section,shape)

        if ravel:
            return sample.ravel()
        else:
            return sample

    def generate_matrix(self, l_image):
        """
            crops resizes and ravels the image 
        """
        labeled_image = l_image
        regions = self.__get_objects_coordinates(labeled_image)
        np_matrix = np.empty((len(regions),400),dtype=np.uint8)
        counter = 0
        for region in regions:
            ratio = (region[1] - region[0])/(region[3] - region[2]) 
            rectangle_area = (region[1] - region[0])*(region[3] - region[2]) 
            if ratio < 1 - ObjectDetector.RECTANGLE_TOLERANCE or ratio > 1 + ObjectDetector.RECTANGLE_TOLERANCE:
                continue
            if rectangle_area > 2000:
                continue
            img_sec = self.crop_resize_ravel(region)
            np_matrix[counter,:] = img_sec
            counter += 1
        return np_matrix[:counter, :].copy()

    def get_object_detected_image(self, l_image):
        """
            Colorizes the image and puts a rectangle around the different zones 
            considered to be noisy. 
        """
        label_image = l_image
        image_label_overlay = label2rgb(label_image, image=self.img_src)
        for region in measure.regionprops(label_image):
            if region.area > self.get_image_area() * 0.05:
                continue
            minr, minc, maxr, maxc = region.bbox
            ratio = (maxr-minr) / (maxc -minc)
            rectangle_area = (maxr-minr)*(maxc-minc)
            if ratio < 1 - ObjectDetector.RECTANGLE_TOLERANCE or ratio > 1 + ObjectDetector.RECTANGLE_TOLERANCE:
                continue
            if rectangle_area > 2000 :
                continue
            rr, cc = polygon_perimeter(r = [minr,minr,maxr,maxr], c = [minc,maxc,maxc,minc],shape =self.img_src.shape)
            image_label_overlay [rr,cc] = (1,0,0)
        return image_label_overlay

    def patch(self, l_image, margin = 5):
        """
            Puts a black rectangle over the regions considered to be noisy to the further analisys.
        """
        label_image = l_image
        patched_image = self.img_src.copy()
        for region in measure.regionprops(label_image):
            if region.area > self.get_image_area() * 0.05:
                continue
            minr, minc, maxr, maxc = region.bbox
            ratio = (maxr-minr) / (maxc -minc)
            rectangle_area = (maxr-minr)*(maxc-minc)
            if ratio < 1 - ObjectDetector.RECTANGLE_TOLERANCE or ratio > 1 + ObjectDetector.RECTANGLE_TOLERANCE:
                continue
            if rectangle_area > 2000 :
                continue
            rr, cc = polygon(r = [minr-margin,minr-margin,maxr+margin,maxr+margin], c = [minc-margin,maxc+margin,maxc+margin,minc-margin],shape =self.img_src.shape)
            patched_image[rr,cc] = 0
        return patched_image
    
    def get_super_cluster(self,patched_image = None, clusterer_type = 'kmeans'):
        clustering_type = {
            'kmeans': KMeans,
            'dbscan': dbscan,
            'aglomerative': AgglomerativeClustering

        }
        patched_image = dilation(patched_image)
        patched_image = resize(patched_image,(512,1024))
        withe_pixels = np.column_stack(np.where(patched_image > 0))
        print(withe_pixels.shape)
        #knn_graph = kneighbors_graph(withe_pixels, 30, include_self=True)
        #clusterer = AgglomerativeClustering(n_clusters=4, linkage='average', connectivity=None, affinity="manhattan")
        clusterer = KMeans(n_clusters=400, precompute_distances=True)
        labels = clusterer.fit_predict(withe_pixels)
        return withe_pixels, labels, patched_image, clusterer.cluster_centers_
        


def process_bn_image(file_path,**kwargs):
    """
        Proccess to be applied to a single image includes loading the image, getting the labels 
        of  different regions, filtering the desired ones, create a region labeled image and deleting 
        the regions considered to be noisy.
    """
    matrix_path = kwargs.get('matrix_path')
    image_path = kwargs.get('image_path')
    patch_dir = kwargs.get('patch_dir')
    clusters_path = kwargs.get("clusters_path")
    object_detector = ObjectDetector(file_path)
    regions_path = file_path[:-3] + 'npy'
    if regions_path :
        try:
            labeled_image = np.load(regions_path)
        except IOError:
            labeled_image = object_detector.object_detect()
            np.save(regions_path,labeled_image)
        
    if matrix_path:
        object_detected_matrix = object_detector.generate_matrix(labeled_image)
        np.save(matrix_path,object_detected_matrix)
    
    if image_path:
        imsave(image_path,object_detector.get_object_detected_image(labeled_image))

    if clusters_path:
        patched = object_detector.patch(labeled_image)
        if patch_dir:
            imsave(patch_dir,patched)
        coordinates, labels, small_patched, centers  = object_detector.get_super_cluster(patched)
        labels += 1
        l_im = patched.copy()
        for label in np.unique(labels):
            cord = coordinates[labels == label, :]
            l_im[cord[:,0],cord[:,1]] = label  
        l_im = label2rgb(l_im,small_patched)
        for center in centers:
            rr, cc = circle(r = center[0], c = center[1], radius = 20, shape =  small_patched.shape)
            l_im[rr,cc] = (0,0,1)
        imsave(clusters_path,l_im)
    elif patch_dir:
        imsave(patch_dir,object_detector.patch(labeled_image))

    

    
def validate_flag(flag_dict, flag_name, flag_value):
    if not isdir(flag_value):
        print('Error: '+ o_dir + " is not a directory")
        exit(1)
    flag_dict[flag_name]=flag_value


''' Main process pipeline '''

if __name__ == "__main__":
    flags = {}
    args = parser.parse_args()
    s_dir = args.file_path
    o_dir = args.outdir
    im_dir = args.imagedir
    p_dir = args.patchdir
    c_dir = args.clusterdir
    if s_dir and isdir(s_dir):
        if s_dir[-1] != '/':
            s_dir += '/'
        elements = listdir(s_dir)
        elements = filter(lambda x: x[-3:] == 'png', elements)
        if o_dir:
            validate_flag(flags,'matrix_path', o_dir)
        if p_dir:
            validate_flag(flags,'patch_dir',p_dir)
        if im_dir: 
            validate_flag(flags,'image_path',im_dir)
        if c_dir:
            validate_flag(flags,'clusters_path',c_dir)

        for elem in elements:
            print('Processing: ' + elem)
            instance_flags = dict(flags)
            if o_dir:
                instance_flags['matrix_path'] += elem[:-4]
            if p_dir:
                instance_flags['patch_dir'] += elem
            if im_dir: 
                instance_flags['image_path'] += elem[:-4] + '_labeled.png'
            if c_dir:
                instance_flags['clusters_path'] += elem
            process_bn_image(s_dir + elem, **instance_flags)
    else:
        object_detector = ObjectDetector(args.file_path)
        try:
            labeled_image = np.load(args.file_path[:-3] + 'npy')
        except IOError:
            labeled_image = object_detector.object_detect()
            np.save(regions_path,labeled_image)
        object_detected_matrix = object_detector.generate_matrix(labeled_image)
        if args.matrix:
            np.save(args.matrix,object_detected_matrix)
        if args.imagepath:
            imsave(args.imagepath,object_detector.get_object_detected_image(labeled_image))
        if args.patch:
            imsave(args.patch, object_detector.patch(labeled_image))
        if args.cluster:
            image = object_detector.patch(labeled_image)
            coordinates, labels, small_patched, centers  = object_detector.get_super_cluster(image)
            labels += 1
            l_im = small_patched.copy()
            for label in np.unique(labels):
                cord = coordinates[labels == label, :]
                l_im[cord[:,0],cord[:,1]] = label  
            l_im = label2rgb(l_im,small_patched)
            print(l_im[0,0])
            for center in centers:
                rr, cc = circle(r = center[0], c = center[1], radius = 2, shape =  small_patched.shape)
                l_im[rr,cc] = (0,0,1)
            imsave(args.cluster,l_im)