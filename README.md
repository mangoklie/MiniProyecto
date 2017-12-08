# Mini proyecto de desarrollo de software

## Requerimientos

1. Python 3.6
2. El conjunto de software WFDB Applications
3. GFortran (Si usas ubuntu ejecutar: ```sudo apt-get install gfortran```)
4. ECGPUWAVE instalado [Presiona aqui para una mayor información](https://physionet.org/physiotools/ecgpuwave/src/)
5. Las bibliotecas señaladas en el **pipfile.txt** haciendo ``` pip install -r pipfile.txt ```

## Transformación se imagenes a señal

Para realizar transformar una imagen a señal utilizar el siguiente comando:
``` python imageProcess.py <Ruta del archivo> <Ruta del archivo de salida> -cpar <Parametro de ajuste de contraste (Numérico)> ```

Si se va a realizar un trabajo por lotes no es necesario el parámetro **-cpar** pero si es necesario definir un archivo de configuración para el ajuste de contraste de los archivos en formato JSON dentr del directorio donde se encuentran las imágenes.

```javascript 
    [
        {
            "name": "img_85.jpg",
            "adjust_param": 0.45
        },
        ...
        {
            "name": "img_Y.jpg",
            "adjust_param": .60
        }
    ]
```
En caso de realizar trabajos por lotes ejecutar el comando: ``` python imageProcess.py <Ruta del archivo> <Ruta del archivo de salida>```

Para mayor detalle de las opciones ejecutar: ``` python imageProcess.py --help ```

## El archivo SignalRerieverClass.py

El archivo contiene una clase encargada de realizar la transformación de imagen a color a imagen binaria (Blanco y/o negro). Si se ejecuta como un script realizará la transformación por lote de los archivos en el directorio *Imagenes*.

## El archivo ObjectDetector.py

Contiene una clase encargada de eliminar (en lo posible) el texto de las imágenes. También se encarga de obtener las coordenadas de los pixeles que representan la señal. Al ejecutar ```python ObjectDetector.py --help````se obtendra una lista de las posibles opciones que el script ofrece. 

* 'file_path', "Ruta del archivo o de los archivos a transformar"
* '-i', '--imagepath',  "Guardar las imagenes etiquetadas en un directorio"
* '-imd', '--imagedir',  "Directorio donde colocar la imagen etiquetada" 
* '-p', '--patch',  "Realiza el cubrimiento de lar regiones consideradas ruidosas (texto)"
* '-pd', '--patchdir',  "Ubicación de las imagenes ajustadas"
* '-e', '--expand',  "Expandir la señal cuando se haga el ajuste"
* '-c','--cluster',  "Una vez ajustadas realizar agrupamieno de los píxeles que representen la señal (Toma tiempo)"
* '-cd', '--clusterdir',  "Directorio donde guardar los pixeles seleccionados"
* '--use_patched',  "Al realizar agrupamiento esta opción carga las imagenes que se encuentren en patch dir (deben coincidir los nombres"

## El archivo SignalRetriever

Contiene una clase encargada de transformar los archivos de puntos obtenidos en el agrupamiento para generar la señal. El script realiza las transformaciones de las imagenes en la señal especificadas y si hay un directorio de salida los debería colocar en este. Ejecutar ```python SignalRetriever.py --help``` para mayor información.

## El archivo SignalProcessor

Contiene una clase encargada de extraer todos los atributos de una señal. Adicionalmente contiene un Script que recibe el nombre de una señal, la procesa y devuelve los atributos obtenidos en la consola o en algún archivo de salida

Es importante decir que el Script se debe ejecutar en el mismo directorio donde se encuentran los archivos de las señales pues de lo contrario NO FUNCIONARÁ. Esto se debe a que la biblioteca wfdb busca los archivos en directorios específicos y no se implementó esta modalidad. 

Para ayuda sobre las opciones que este script ofrece ejecute ``` python SignalProcessor.py --help```.

Para extraer los atributos de los archivos de todo un directorio realizar:

```bash
cd /ruta/al/directorio/de/los/arcchivos
python /ruta/al/script/SignalProcessor.py -f ./ -o /ruta/al/archivo/de/salida
```

NOTA: El archivo REFERENCE-v3.csv debe estar en la carpeta padre de donde residen los archivos de señales.



