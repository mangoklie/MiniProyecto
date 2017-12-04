import argparse
from ObjectDetector import ObjectDetector, process_bn_image
from SignalRerieverClass import ImageSignalDigitalizer
from SignalRetriever import SignalRetriever
from skimage import img_as_uint
from skimage.io import imread, imsave
from os import mkdir
from os.path import isdir, isfile, exists
from sys import stderr

parser = argparse.ArgumentParser()
# Arguments for the script
parser.add_argument('file_path', help ="The path where the file(s) are")
parser.add_argument('output_dir', help = "The path where the final signal files will be")
parser.add_argument('-bp', '--bw_path', help = "The path where black and withe images will be stored")
parser.add_argument('-ip','--image_path', help = "The path where labeled images will be stored")
parser.add_argument('-cpar','--contrast_param', help = 'Contrast parameter to process the image')
parser.add_argument('-pd','--patch_dir', help = "The path where the patched images will be")
parser.add_argument('-lp','--load_patched', help = "The filepath where the patched images are", default = False, action = "store_true")
parser.add_argument('-cp','--cluster_path', help = "The path where the desired pixel points will be")

def bw_process(file, bw_path, param = .85):
    if '.json' in file:
        image_digitalizer = ImageSignalDigitalizer(config_file = file)
        
        if bw_path:
            if bw_path[-1] != '/':
                bw_path += '/'
        else:
            bw_path = './bn_images/'
            if not isdir(bw_path):
                mkdir(bw_path)

        temp_name = bw_path + 'bn_{0}.png'
        for x in image_digitalizer.process_images():
            image = img_as_uint(x[0])
            imsave(temp_name.format(x[1][:-4]), image)
            yield(image,x[1])
    else:
        last_name_file = file.split('/')[-1]
        image = imread(file)
        image = ImageSignalDigitalizer.process_image(image,param)
        image = img_as_uint(image)
        if bw_path:
            if bw_path[-1] != '/':
                bw_path += '/'
            temp_name = bw_path + 'bn_{0}.png'
            imsave(temp_name.format(last_name_file[:-4]), image)
        yield image, 'bn_'+file.split('/')[-1][:-4]+'.png'

def object_detect(*args,**kwargs):
    file_path = kwargs.get('bn_path',None)
    print(kwargs)
    if file_path and isfile(file_path):
        object_detector = ObjectDetector(file_path)
    else:
        assert len(args) < 2 and len(args) > 0, 'Only one file argument'
        object_detector = ObjectDetector()
        object_detector.img_src = args[0]
    process_bn_image(file_path,**kwargs)

def signal_create(file_path,output_dir):
    files = None
    if file_path:
        if isdir(file_path):
            files = map(lambda x: file_path+x ,filter(lambda x: re.match('.*\.npy',x),listdir(file_path)))
        else:
            s_retriever = SignalRetriever(file = file_path)
            record = s_retriever.get_record_signal(output_dir)
            #record.wrsamp()

    if files:
        for image_sample in files:
            print("Processing coordinates: " + image_sample)
            s_retriever = SignalRetriever(image_sample)
            record = s_retriever.get_record_signal(output_dir)
            #record.wrsamp()

def handle_dirs(output_dir, default_name):
    if not output_dir:
        try:
            mkdir(default_name)
        except FileExistsError:
            pass
        return default_name
    elif exists(output_dir):
        if isfile(output_dir):
            raise FileExistsError("Output directory is a regular file")
        elif not isdir(output_dir):
            raise FileExistsError("Output directory is not a directory file")
        else:
            return output_dir
    else:
        mkdir(output_dir)
        return output_dir
        

if __name__ == "__main__":
    args = parser.parse_args()
    file_path = args.file_path
    output_dir = handle_dirs(args.output_dir,'./signals_dir/')
    images_path = handle_dirs(args.image_path, './label_path/')
    bw_path = handle_dirs(args.bw_path,'./bw_dir/')
    patch_dir = handle_dirs(args.patch_dir,'./patch_dir/')
    load_patched = args.load_patched
    cluster_path = handle_dirs(args.cluster_path,'./cluster_dir/')
    cparam = args.contrast_param
    if cparam is None:
        items_iterator = bw_process(file_path,bw_path)
    else:
        items_iterator = bw_process(file_path,bw_path, float(cparam))
    for image_item in items_iterator:
        image, name = image_item
        object_detect(bn_path = bw_path+name, patch_dir = patch_dir+name, clusters_path = cluster_path+name, image_path = images_path + name )
        signal_create(cluster_path + name[:-3]+'npy', output_dir + name)

    
    