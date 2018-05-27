# Face_RecognitionOpenCv2
Reconocimiento Facial en tiempo real con Python y OpenCV

# Pasos

# Intallar OpenCV

- pip install opencv-python
- pip install opencv-contrib-python


# Crear dataset para entrenamiento

* En la carpeta images crear una carpeta por persona. Ejemplo si quiero almacenar imagenes de Eduardo
  debo crear una carpeta dentro de images con el nombre Eduardo la ruta quedaría images/Eduardo

* Modificar la ruta en donde se guardarán las fotos del rostro de la persona que está parada frente a la web-cam y modificiar el nombre   que tendrán las fotos, por convención estás fotos es mejor llamarlas como se llama la persona y como se llama la carpeta.
  Esta ruta se modifica en el archivo de python dataset.py en la linea 21.

  cv2.imwrite("images/Eduardo/Eduardo_"+str(count)+".jpg", grises[y:y+h, x:x+w])

* Entrar desde la consola a la ruta raíz del proyecto y ejecutar con: # python dataset.py

* Verificar que las imágenes creadas por dataset.py estén en la carpeta indicada en el paso anterior.

* La carpeta images ya contiene algunas imágenes, si va a utilizarlas debe descomprimir los archivos zip dentro de cada carpeta

# Crear archivo entrenamiento.yml y labels.pickle

* Ejecutar el script de python llamado entrenamiento.py al realizar esto se crearán ambos archivos en la raíz del proyecto


# Probar y detectar rostros en tiempo real

* Ejecutar el script de python llamado detectar_identificar_rostros.py


  


