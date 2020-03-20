#!/usr/bin/python
# encoding: utf-8

import time
import subprocess
import picamera
import requests
from gpiozero import led
from datetime import datetime

requests.urllib3.disable_warnings()

DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")[:-3]

ledRojo = led(24)
ledAzul = led(23)
ledVerde = led(18)


def tomarFoto():
    ledAzul.off()
    picam.resolution = (640, 480)
    picam.framerate = 32
    picam.ISO = 600
    picam.brightness = 60
    picam.rotation = 180
    picam.capture('talcual.jpg')
    pass


def leerTarjeta():
    try:
        lecturaTarjeta = subprocess.Popen(['/home/pi/libnfc-1.7.1.3/examples/nfc-poll'], stdout=subprocess.PIPE)
        lecturaTarjeta.wait()
        llegada = lecturaTarjeta.stdout.read().decode('utf-8')
        temp = []
        t = 0
        j = 0
        while j in range(14):
            if j > 1:
                if j % 2 == 0:
                    j = j + 2

            temp[t] = llegada[j]
            t = t + 1
            j = j + 1
        return temp
    except:
        KeyboardInterrupt()


def envioDatos(idTarjeta):
    url = 'https://rita.udistrital.edu.co/sistema-acceso/upload'
    file = {'photo': open('/home/pi/talcual.jpg', 'rb')}
    datai = {'id': idTarjeta, 'tipo': 'Entrada', 'dispositivo': 'Prueba1'}
    print(datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")[:-3] + "   " + str(datai))
    try:
        print(datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")[:-3] + "   Sending")
        r = requests.post(url, data=datai, files=file, verify=False)
        print(r.status_code)
        print(datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")[:-3])
        return r.status_code

    except requests.ConnectionError as e:
        print(datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")[:-3] + "   " + "Problemas de Conexion con el servidor")
        return (403)


def validarRespuesta(respuesta):
    if respuesta == 200:
        ledVerde.on()
        time.sleep(1)
        ledVerde.off()
        ledAzul.on()

    else:
        ledRojo.on()
        time.sleep(1)
        ledRojo.off()
        ledAzul.on()


if __name__ == "__main__":
    llave = 0
    ledRojo.on()
    ledAzul.on()
    ledVerde.on()
    time.sleep(2)
    ledRojo.off()
    ledVerde.off()
    try:
        while llave == 0:
            nulo = [0] * 8
            identificadorTarjeta = leerTarjeta()
            if (identificadorTarjeta == nulo):
                print("No se encontró tarjeta")
            else:
                print(identificadorTarjeta)
                with picamera.PiCamera() as picam:
                    tomarFoto()
                    picam.close()
                validarRespuesta(envioDatos(identificadorTarjeta))

    except:
        KeyboardInterrupt()
