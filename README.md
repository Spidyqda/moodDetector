# Implementación de una IA para detectar el estado de ánimo de las personas con el fin de calificar el rendimiento de servicio al cliente de las empresas

Este repositorio contiene el código fuente y la documentación de un proyecto de detección de estado de ánimo en tiempo real utilizando un modelo de deep learning, que funcione como API REST.

## Descripción del Proyecto

El proyecto tiene como objetivo proporcionar a las empresas una herramienta para medir la satisfacción del cliente a través de la detección de emociones en tiempo real. Utiliza un modelo de aprendizaje profundo basado en redes neuronales convolucionales (CNN) para analizar las expresiones faciales y asignar una calificación de satisfacción. El aplicativo web, esta escrito en el micro-framework Flask y trae consigo una demo de lo aplicacion.

## Características Principales

- Detección en tiempo real de emociones a través de la cámara.
- Implementación de un modelo de aprendizaje profundo para análisis de expresiones faciales.
- Sistema de calificación universal basado en letras para la satisfacción del cliente.
- Interfaz web para visualización de resultados.

## Estructura del Repositorio

- `PaginaWeb/`: Contiene el código fuente del proyecto.
- `PaginaWeb/static/`: Almacena archivos estáticos, en este caso las hojas de estilo.
- `PaginaWeb/templates/`: Contiene archivos HTML para la interfaz web.
- `PaginaWeb/model/`: Contiene la arquitectura del modelo en JSON junto a sus pesos.
 	(Se puede descargar los pesos del modelo en la seccion de releases)
- `PaginaWeb/images/`: Contiene imagenes de prueba con las que se puedo probar el modelo.

- `ModeloML/`: Contiene la implementacion del modelo CNN, su entrenamiento y las pruebas realizadas para observar su rendimiento.

- `/`: Contiene la pagina web dentro de la carpeta "PaginaWeb/", el modelo entrenado en la carpeta "ModeloML/" y los requerimientos en el archivo "requirements.txt".

## Instrucciones de Uso

1. Clona este repositorio: `git clone https://github.com/tu_usuario/tu_proyecto.git`
2. Instala las dependencias: `pip install -r requirements.txt`
3. Ejecuta la aplicación: `python app.py`
4. Accede a la interfaz web desde tu navegador: `http://localhost:5000`
### Una vez dentro de la pagina
5. Puedes subir imagenes para probar la eficacia del modelo en la pagina "demostracion.html"
6. Al abrir la pagina "Camara.html" la camara se iniciara
7. Para iniciar la prueba, presionar el boton "Iniciar Prueba"
8. Para finalizar la prueba, presionar el boton "Finalizar Prueba"
9. Para observar los resultados, presionar el boton "Mostrar Resultados"

## Contribuciones

¡Las contribuciones son bienvenidas! Si encuentras algún problema o tienes sugerencias, abre un problema o envía una solicitud de extracción.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para obtener más detalles.
