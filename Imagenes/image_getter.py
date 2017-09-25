
from bs4 import BeautifulSoup
import requests as rq
from io import BytesIO
import skimage.io as sio

PARSER = "html.parser"
SOURCE_FILE = 'table_url_images.html'

if __name__ == "__main__":
    html_file = open(SOURCE_FILE,'r')
    out_csv = open('pathology_file.csv','w')
    soup = BeautifulSoup(html_file,PARSER)
    
    #Printing the header of the csv file
    print('filename,pathology',file=out_csv)
    
    #Gettring urls and descriptions
    elem_lists = []
    i=0
    for elem  in soup.select('tr'):
        tuple_elems = elem.select('td.column-1 > a, td.column-2')
        
        try:
            try:
                tuple_elems[0] = tuple_elems[0]['href']
                try:
                    r = rq.get(tuple_elems[0])
                    image_name = 'img_{0}.jpg'.format(i)
                    i+=1
                    image = sio.imread(BytesIO(r.content))
                    sio.imsave(image_name,image)
                    tuple_elems[0]=image_name
                except Exception as e:
                    print(e)    
            except KeyError:
                tuple_elems[0] = '?'
            try:
                tuple_elems[1] = tuple_elems[1].contents[0].replace(',',' -')
            except Exception:
                tuple_elems[1] = '??'
        except IndexError:
            tuple_elems.insert(0,'Not Found')
        print('{0},{1}'.format(*tuple_elems),file=out_csv)
    
    #Closing files
    html_file.close()
    out_csv.close()
