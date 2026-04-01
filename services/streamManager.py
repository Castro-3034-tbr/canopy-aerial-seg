import logging
import multiprocessing
from uuid import uuid4

from fastapi import HTTPException

from data.projectData import initProjectData
from data.sharedData import initSharedData
from services.procesorProcess import processorProcess
from services.readerProcess import readerProcess


class StreamManager:
    """Gestiona múltiples sesiones RTSP en paralelo."""

    def __init__(self, manager, model_config: dict, save_path_config: dict) -> None:
        """Inicializa el StreamManager con la configuración del modelo y las rutas de guardado.

        Args:
            manager: Instancia del Manager para crear namespaces y colas compartidas entre procesos.
            model_config (dict): Configuración del modelo YOLO, incluyendo la ruta del modelo y el dispositivo a usar.
            save_path_config (dict): Configuración de las rutas para guardar logs e inferencias, con claves "Logs" e "Inference" respectivamente.
        """
        self.manager = manager                              # Instancia del Manager para crear namespaces y colas compartidas entre procesos
        self.model_config = model_config                    # Configuración del modelo YOLO, incluyendo la ruta del modelo y el dispositivo a usar
        self.save_path_config = save_path_config            # Configuración de las rutas para guardar logs e inferencias, con claves "Logs" e "Inference" respectivamente
        self.sessions: dict[str, dict] = {}                 # Diccionario para almacenar la información de cada sesión activa, con streamId como clave y un diccionario con detalles de la sesión como valor

    def start(
        self,
        stream_id: str | None,
        rtsp_url: str,
        save_log: bool,
        save_inference: bool,
        confidence_class: float,
        mqtt_broker: str,
        mqtt_port: int,
        mqtt_topic: str,
    ) -> dict:
        """Inicia una nueva sesión de procesamiento de un stream RTSP con la configuración especificada.

        Args:
            stream_id (str | None): _description_
            rtsp_url (str): _description_
            save_log (bool): _description_
            save_inference (bool): _description_
            confidence_class (float): _description_
            mqtt_broker (str): _description_
            mqtt_port (int): _description_
            mqtt_topic (str): _description_

        Raises:
            HTTPException: _description_
            HTTPException: _description_
            HTTPException: _description_

        Returns:
            dict: _description_
        """
        #Generamos un streamId único si no se proporcionó uno, y verificamos que no exista una sesión con el mismo streamId
        session_id = stream_id or f"stream-{uuid4().hex[:8]}"
        if session_id in self.sessions:
            raise HTTPException(
                status_code=409,
                detail=f"Ya existe una sesión con streamId '{session_id}'.",
            )
        
        #Creacion de los namespaces y colas compartidas entre procesos, y configuración de los procesos de lectura y procesamiento del stream
        shared_data = initSharedData(self.manager)
        project_data = initProjectData(
            self.manager,
            self.model_config.get("Path", "yolov8n-seg.pt"),
            self.model_config.get("Device", "cpu"),
            self.save_path_config.get("Logs", "logs/"),
            self.save_path_config.get("Inference", "inference/"),
        )
        
        #Inicializacion de los flgs de control de los procesos de lectura
        project_data.readerProcessRunning = True
        project_data.processorProcessRunning = True

        #Creacion de los procesos de lectura y procesamiento del stream
        reader_process = multiprocessing.Process(
            target=readerProcess,
            args=(shared_data, project_data, rtsp_url),
            name=f"ProcesoReader-{session_id}",
            daemon=True,
        )
        processor_process = multiprocessing.Process(
            target=processorProcess,
            args=(
                shared_data,
                project_data,
                save_log,
                save_inference,
                confidence_class,
                mqtt_broker,
                mqtt_port,
                mqtt_topic,
            ),
            name=f"ProcesoProcessor-{session_id}",
            daemon=True,
        )

        #Intento de iniciar los procesos y verificacion de que se encuentran activos 
        try:
            reader_process.start()
            processor_process.start()
        except Exception as e:
            #En caso de error, se reinician los flasgs de control y se envia un mensaje de error con el detalle del problema
            project_data.readerProcessRunning = False
            project_data.processorProcessRunning = False
            logging.exception("No se pudieron iniciar los procesos del stream")
            raise HTTPException(
                status_code=500,
                detail="No se pudieron iniciar los procesos del stream.",
            ) from e
            
        #Si los procesos no quedaron activos tras el intento de inicio, se actualizan los flags de control y se lanza una excepción
        if not reader_process.is_alive() or not processor_process.is_alive():
            project_data.readerProcessRunning = False
            project_data.processorProcessRunning = False
            raise HTTPException(
                status_code=500,
                detail="Los procesos del stream no quedaron activos tras el arranque.",
            )

        #Guardado de la informacion de la sesion en el directorio de sensiones activas
        self.sessions[session_id] = {
            "stream_id": session_id,                        #Identificador unico de la sesion
            "state": "running",                             #Estado actual de la sesion, inicialmente "running"
            "rtsp_url": rtsp_url,                           #URL del stream RTSP que se esta procesando en esta sesion
            "shared_data": shared_data,                     #Namespace compartido de sharedData
            "project_data": project_data,                    #Namespace compartido de projectData 
            "reader_process": reader_process,               #Proceso de lectura del stream RTSP
            "processor_process": processor_process,         #Proceso de procesamiento del stream RTSP
            "mqtt": {                                       #Informacion de configuracion MQTT
                "broker": mqtt_broker,
                "port": mqtt_port,
                "topic": mqtt_topic,
            },
        }

        #Retorno de la informacion relevante de la sesion iniciada
        return {
            "msg": "Inicio del stream realizado correctamente",
            "streamId": session_id,
            "state": "running",
            "rtspUrl": rtsp_url,
            "mqtt": self.sessions[session_id]["mqtt"],
        }

    
    def stop(self, stream_id: str | None = None, timeout: float = 5.0) -> dict:
        """Detiene un sesion de procesamiento de un stream RTSP, para 
        eso es necesario identificar la sesion a detener mediante su streamId, 
        si no se proporciona un streamId, se detendran todas las sesiones activas.

        Args:
            stream_id (str | None, optional): identificador de la sesion a detener. Defaults to None
            timeout (float, optional): tiempo máximo de espera para terminar los procesos. Defaults to 5.0.

        Raises:
            HTTPException: Si no existe una sesión con el streamId proporcionado.

        Returns:
            dict: Un diccionario con un mensaje de resultado y la información relevante de la sesión detenida o de las sesiones detenidas.
        """
        
        #Comprobacion de que se proporciono un streamId, si no se proporciono, se detienen todas las sesiones activas y se retorna un mensaje con la informacion de las sesiones detenidas
        if stream_id is None:
            stopped_streams = [
                self._stopOne(session_id, timeout)
                for session_id in list(self.sessions)
            ]
            return {
                "msg": "Detención de streams realizada correctamente",
                "stopped": stopped_streams,
            }

        #Comprobacion de que existe una sesion con el streamId proporcionado
        if stream_id not in self.sessions:
            raise HTTPException(
                status_code=404,
                detail=f"No existe ninguna sesión con streamId '{stream_id}'.",
            )

        #Detencion de la sesion identificada por el streamId proporcionado
        stopped = self._stopOne(stream_id, timeout)
        return {
            "msg": "Detención del stream RTSP realizada correctamente",
            "streamId": stopped["streamId"],
            "state": stopped["state"],
        }

    def health(self) -> dict:
        """ Exponen el estado actual de todas las sesiones activas

        Returns:
            dict: Informacion del estado actual de los streams activos
        """
        return {
            "active_streams": len(self.sessions),
            "streams": [
                {
                    "streamId": session_id,
                    "state": session["state"],
                    "rtspUrl": session["rtsp_url"],
                    "reader_alive": bool(session["reader_process"].is_alive()),
                    "processor_alive": bool(session["processor_process"].is_alive()),
                }
                for session_id, session in self.sessions.items()
            ],
        }

    def _stopOne(self, stream_id: str, timeout: float) -> dict:
        """Detiene una sesión de procesamiento de un stream RTSP identificada por su streamId, 
        esperando a que los procesos de lectura y procesamiento terminen correctamente, 
        y si no lo hacen en el tiempo especificado, se fuerza su terminación.

        Args:
            stream_id (str): identificador de la sesión a detener
            timeout (float): tiempo máximo de espera para que los procesos terminen correctamente antes de forzar su terminación

        Returns:
            dict: información sobre la sesión detenida
        """
        
        #Actualización de los flags de control para indicar a los procesos que deben detenerse
        session = self.sessions[stream_id]
        session["state"] = "stopping"
        session["project_data"].readerProcessRunning = False
        session["project_data"].processorProcessRunning = False

        #Espera para que los procesos de lectura y procesamiento terminen correctamente
        for process in (session["reader_process"], session["processor_process"]):
            #Espera a que el proceso termine correctamente dentro del tiempo de espera especificado
            process.join(timeout=timeout)
            
            #Forzado de terminacion del proceso si sigue activo después del tiempo de espera
            if process.is_alive():
                logging.warning(
                    "El proceso %s no terminó a tiempo; se fuerza terminate().",
                    process.name,
                )
                process.terminate()
                process.join(timeout=timeout)

        #Eliminacion de la sesion del directorio de sesiones activas y retorno de la informacion relevante de la sesion detenida
        del self.sessions[stream_id]
        return {"streamId": stream_id, "state": "stopped"}
