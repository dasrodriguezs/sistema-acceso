import datetime
import os
import uuid

from flask import Flask, request, jsonify
from static.code.identificador_rostros import Identificador
from static.code.logger_handler import get_logger

app = Flask(__name__)
identificador = Identificador('')

logger = get_logger(__name__)


@app.route("/sistema-acceso/upload", methods=['POST'])
def upload():
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':
            location = os.path.join('static', 'images', str(uuid.uuid4()) + photo.filename)
            photo.save(location)
            identificador.img_dir = location
            logger.debug('ID: ' + request.form.get('id'))
            mensaje = identificador.identify(request.form.get('id'), request.form.get('tipo'),
                                             request.form.get('dispositivo'))
            aut = {'list': []}
            for obj in mensaje.get('list'):
                if obj.get('proba') > 0.01 and obj.get('autorizado') is True:
                    aut.get('list').append(obj)
            if len(aut.get('list')) > 0:
                return jsonify(aut), 200
            else:
                return jsonify(aut), 403
        return jsonify({'error': 'no photo provided'}), 400
    return jsonify({'error': 'no image provided'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
