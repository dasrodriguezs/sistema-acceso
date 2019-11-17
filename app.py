import datetime
import os
import uuid

from flask import Flask, request, jsonify
from static.code.identificador_rostros import Identificador

app = Flask(__name__)
algo = Identificador('')


@app.route("/upload", methods=['POST'])
def upload():
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':
            location = os.path.join('static', 'images', str(uuid.uuid4()) + photo.filename)
            photo.save(location)
            algo.img_dir = location
            print(str(datetime.datetime.utcnow()) + ' ID: ' + request.form.get('id'))
            mensaje = algo.identify(request.form.get('id'), request.form.get('tipo'), request.form.get('dispositivo'))
            aut = {'list': []}
            for obj in mensaje.get('list'):
                if obj.get('proba') > 0.5 and obj.get('autorizado') is True:
                    aut.get('list').append(obj)
            if len(aut.get('list')) > 0:
                return jsonify(aut), 200
            else:
                return jsonify(aut), 403
        return jsonify({'error': 'no photo provided'}), 400
    return jsonify({'error': 'no image provided'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
