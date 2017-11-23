from ObjectDetector import ObjectDetector
from SignalRerieverClass import ImageSignalDigitalizer, imread, np
from SignalProcessor import SignalProcessor, get_features
from SignalRetriever import SignalRetriever

def transform_image(image,duration=30, dummy_image = './foo.png', temp_file = 'temp'):
    """
        Given a image file it returns the corresponding Signal processor object with 
        the segments already detected. Once the image is loaded (and assuming it doesn't have any noise)
        the function should return an object to get the desired features. 
    """
    o_detector = None
    if isinstance(image,str):
        o_detector = ObjectDetector(image)
    elif isinstance(image, np.ndarray):
        o_detector = ObjectDetector(dummy_image)
        o_detector.img_src = image
    
    transformation_image = ImageSignalDigitalizer.bw_transform_special(o_detector.img_src)
    coordinates = o_detector.get_super_cluster(transform_image)
    retriever_object = SignalRetriever(array = coordinates, arr_name = temp_file)
    retriever_object.get_record_signal()

    processor = SignalProcessor(temp_file)
    processor.detect_segments_new()
    return processor



