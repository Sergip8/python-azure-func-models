
import base64
from io import BytesIO
import json
import azure.functions as func
import logging
from PIL import Image
from traslate_manga_image import MangaTranslator

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Obtener los datos de la imagen desde la petición POST
        image_data = req.files.get("image_data")


        if not image_data:
            return func.HttpResponse(
                "No file found in the request. Please upload a file with key 'image'.",
                status_code=400
            )
            
        # Get the filename from the uploaded file
        filename = image_data.filename
        logging.info(filename)
        # Read the file content
        file_content = image_data.read()
        
       
        # Abrir la imagen con Pillow
        image = Image.open(BytesIO(file_content))
        logging.info(f"Image format: {image.format}")
        processed_image = MangaTranslator().translate_manga_page(image, "es")
        # Ejemplo de procesamiento: convertir a escala de grises
       

        # Guardar la imagen procesada en memoria
        output_buffer = BytesIO()
        processed_image.save(output_buffer, format="JPG")
        processed_image_bytes = output_buffer.getvalue()

        # Encodificar la imagen procesada en base64
        encoded_image = base64.b64encode(processed_image_bytes).decode('utf-8')
        json_response = json.dumps({
            "image": encoded_image,
            "filename": filename
        })
        # Devolver la imagen procesada como respuesta HTTP
        return func.HttpResponse(
            body=json_response,
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error al procesar la imagen: {e}")
        return func.HttpResponse(
            body= e,
            mimetype="application/json",
            status_code=500
        )