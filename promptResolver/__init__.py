
import json
import azure.functions as func
import logging
import spacy
from pymongo import MongoClient
#from dotenv import load_dotenv
import re
import os


#load_dotenv()
filter = {
    "page": 0,
    "size": 16,
    "min": 0,
    "max": 0,
    "areaMin": 0,
    "areaMax": 0,
    "type": [
    ],
    "barrio": [],
    "localidad": None,
    "antiguedad": [],
    "estrato": [],
    "habitaciones": [],
    "banos": [],
    "garajes": [],
    "order": None,
    "operation": "arriendo",
    "location": None
}


def main(req: func.HttpRequest) -> func.HttpResponse:
    
    client = MongoClient(os.getenv('MONGO_URI'))
    logging.info('Python HTTP trigger function processed a request.')
    db = client['LocalsDB']
    collection = db['Locals']

    prompt = req.params.get('prompt')
    if not prompt:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            prompt = req_body.get('prompt')

    if prompt:
        prompt_res = get_nlp_model_response(prompt)
        group_res = agrupar_etiquetas_repetidas(prompt_res)
        procesar_entidades(group_res)
        # docs = buscar_documentos(tag_res, collection)
        # documentos_transformados = []
        # for documento in docs:
        #     documento["_id"] = str(documento["_id"])  # Convertir ObjectId a string
        #     documentos_transformados.append(documento)
        # json_response = json.dumps({"data": documentos_transformados, "query": tag_res, "model": prompt_res}, default=str)
        json_response = json.dumps({"model": group_res, "filter": filter}, default=str)
        return func.HttpResponse(
        body=json_response,
        status_code=200, 
        mimetype="application/json",
        charset="utf-8"  
    )
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
    
def get_nlp_model_response(prompt):
    trained_nlp = spacy.load("localizaloc_model1")
    doc = trained_nlp(prompt)
    return [(ent.text, ent.label_) for ent in doc.ents]

def agrupar_etiquetas_repetidas(entidades):
    """
    Agrupa las entidades con etiquetas repetidas.

    Args:
        entidades (list): Una lista de tuplas (texto, etiqueta).

    Returns:
        list: Una lista de tuplas agrupadas (texto agrupado, etiqueta).
    """

    etiquetas_agrupadas = {}
    resultado = []

    for texto, etiqueta in entidades:
        if etiqueta in etiquetas_agrupadas:
            etiquetas_agrupadas[etiqueta].append(texto)
        else:
            etiquetas_agrupadas[etiqueta] = [texto]

    for etiqueta, textos in etiquetas_agrupadas.items():
        if len(textos) > 1:
            resultado.append((tuple(textos), etiqueta))
        else:
            resultado.append((textos[0], etiqueta))

    return resultado

def procesar_entidades(entidades):
    """
    Procesa las entidades extraídas, validando y extrayendo información específica.

    Args:
        entidades (list): Una lista de tuplas (texto, etiqueta).

    Returns:
        dict: Un diccionario con los datos procesados y validados.
    """

    datos_procesados = {}
    datos_procesados["ubicacion"] = []
    op1 = ["minima", "menor", "mínima"]
    op2 = [ "maxima", "mayor", "máxima"]

    for texto, etiqueta in entidades:
        if etiqueta == "PROPERTY_TYPE":
            if isinstance(texto, tuple):
                for t in texto:
                    if t in ["apartamento", "casa", "local", "finca", "oficina", "bodega", "lote"]:
                        filter["type"].append(t)
            else:      
                if texto in ["apartamento", "casa", "local", "finca", "oficina", "bodega", "lote"]:
                    filter["type"].append(texto)
        elif etiqueta == "OPERATION":
            if texto in ["arriendo", "alquilar"]:
                filter["operation"] = "Arriendo"
            elif texto in ["venta", "comprar"]:
                filter["operation"] = "Venta"
            else:
                filter["operation"] = "Arriendo"
        # elif etiqueta == "LOCATION_TYPE":
        #     if texto in ["barrio", "vereda", "centro", "zona"]:
        #         datos_procesados["tipo_ubicacion"] = texto
        elif etiqueta == "LOCATION":  
            if isinstance(texto, tuple):
                for t in texto:
                    datos_procesados["ubicacion"].append(t)
            else:           
                datos_procesados["ubicacion"].append(texto)
        elif etiqueta == "BEDROOMS":
            if isinstance(texto, tuple):
                for t in texto:
                    numero_habitaciones = re.search(r"\d+", t)
                    if numero_habitaciones:
                        filter["habitaciones"].append(int(numero_habitaciones.group(0)))
                 
            else:       
                numero_habitaciones = re.search(r"\d+", texto)
                if numero_habitaciones:
                    filter["habitaciones"].append(int(numero_habitaciones.group(0)))
                else :
                    numero_habitaciones = re.search(r"^un|una", texto)
                    if numero_habitaciones:
                        filter["habitaciones"].append(1)
        elif etiqueta == "GARAGE":
            if isinstance(texto, tuple):
                for t in texto:
                    numero_garajes = re.search(r"\d+", t)
                    if numero_garajes:
                        filter["garajes"].append(int(numero_garajes.group(0)))
              
            else:
                numero_garajes = re.search(r"\d+", texto)
                if numero_garajes:
                    filter["garajes"].append(int(numero_garajes.group(0)))
                else :
                    numero_garajes = re.search(r"^un|una", texto)
                    if numero_garajes:
                        filter["garajes"].append(1)
             
        elif etiqueta == "AREA_SIZE":
            medida_area = re.search(r"(\d+)\s*?(m2|hectáreas|metros|m)", texto, re.IGNORECASE)
            area = 50
            if medida_area:
                area = int(medida_area.group(1))
            if etiqueta == "AREA_OPERATION":
                for op in op1:
                    if op in texto:
                        filter["areaMin"] = area
                for op in op2:
                    if op in texto:
                        filter["areaMax"] = area
            else:
                filter["areaMin"] = area - area*0.2
                filter["areaMax"] = area + area*0.2

        elif etiqueta == "COST":
            if isinstance(texto, tuple):
                for t in texto:
                    valor_costo = re.search(r"\d+", t)
                    valor = 0
                    if valor_costo:
                        valor = int(valor_costo.group(0))
                        
                      
                        if filter["min"] < valor:
                            filter["min"] = filter["max"] 
                            filter["max"] = valor
                           
                        else:
                            filter["min"] = filter["max"] 
                            filter["max"] = valor
            else:
                valor_costo = re.search(r"\d+", texto)
                valor = 0
                if valor_costo:
                    valor = int(valor_costo.group(0))
                if etiqueta == "COST_OPERATION":
                    for op in op1:
                        if op in texto:
                            filter["min"] = valor
                    for op in op2:
                        if op in texto:
                            filter["max"] = valor
        elif etiqueta == "BATHROOMS":
            if isinstance(texto, tuple):
                for t in texto:
                    numero_banos = re.search(r"\d+", t)
                    if numero_banos:
                        filter["banos"].append(int(numero_banos.group(0)))
            else:
                numero_banos = re.search(r"\d+", texto)
                if numero_banos:
                    filter["banos"].append(int(numero_banos.group(0)))
                else :
                    numero_banos = re.search(r"^un|una", texto)
                    if numero_banos:
                        filter["banos"].append(1)




def buscar_documentos(datos_procesados, collection):
    """
    Busca documentos en MongoDB basados en los datos procesados.

    Args:
        datos_procesados (dict): Diccionario con los datos procesados.
        db_nombre (str): Nombre de la base de datos.
        collection_nombre (str): Nombre de la colección.

    Returns:
        list: Una lista de documentos que coinciden con la consulta.
    """

    query = {}

    if "tipo_propiedad" in datos_procesados:
        query["technicalSheet.property_type_name"] = datos_procesados["tipo_propiedad"]
    if "operacion" in datos_procesados:
        query["operation"] = datos_procesados["operacion"]
    if "ubicacion" in datos_procesados:
        ubicacion = datos_procesados["ubicacion"]
        if len(ubicacion) > 0:
            if len(ubicacion) == 1:
                query["url_frag"] = {"$regex": ubicacion[0], "$options": "i"}
            else:
                regex_pattern = "|".join(datos_procesados["ubicacion"])
                query["url_frag"] = {"$regex": regex_pattern, "$options": "i"}
    if "habitaciones" in datos_procesados:
        query["technicalSheet.bedrooms"] = datos_procesados["habitaciones"]
    if "garajes" in datos_procesados:
        query["technicalSheet.garage"] = datos_procesados["garajes"]
    if "area" in datos_procesados:
        if datos_procesados["operacion_area"] is not None:
            if datos_procesados["operacion_area"] == "minima":
                query["area"] = {"$gte": datos_procesados["area"]["valor"]}
            elif datos_procesados["operacion_area"] == "maxima":
                query["area"] = {"$lte": datos_procesados["area"]["valor"]}
        else:
            area_value = datos_procesados["area"]["valor"]
            query["area"] = {"$lte": area_value + area_value*0.2, "$gte": area_value - area_value*0.2}

    if "costos" in datos_procesados:
        if datos_procesados["operacion_costo"] == "precio máximo":
            query["costs.sale_amount"] = {"$lte": datos_procesados["costos"][0]}
        elif datos_procesados["operacion_costo"] == "valor de arriendo":
            query["costs.rental_amount"] = {"$gte": datos_procesados["costos"][0]}
        else:
            query["costs.rental_amount"] = {"$gte": datos_procesados["costos"][0], "$lte": datos_procesados["costos"][-1]}
    if "banos" in datos_procesados:
        query["technicalSheet.bathrooms"] = datos_procesados["banos"]

    documentos = list(collection.find(query).limit(10))
    return documentos