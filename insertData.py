# -*- coding: utf-8 -*-
"""
This script inserts data into a scheme previously created for outlier detection experiments.

Arguments:
    -c, --config: JSON file with the information required to insert data
    -N, --datasetName: name of the imported dataset
    -f, --fileName: file where data is stored
    -t, --type: Data type, vector (default) or image.

Created on 3/7/2018

@author: Oriol Ramos Terrades (oriolrt@cvc.uab.cat)
@Institution: Computer Vision Center - Universitat Autònoma de Barcelona
"""

import json
import logging
import os
import sys
import struct

try:
    from processData import mongoConnection as mg
    from options import Options
except ImportError:
    from .processData import mongoConnection as mg
    from .options import Options


def deleteDataset(dbms: mg.mongoConnection.startSession, dataset: str):
    """
    Esborra totes les dades de 'dataset' en totes les col·leccions

    :param dbms: connexió a la base de dades
    :param dataset: nom del datase
    :return: True si tots els valors s'han esborrat corectament
    """

    if dataset not in dbms.bd.list_collection_names():  # If collection does not exist, nothing deleted
        return False

    dbms.bd.dataset.drop()  # Collection deleted
    return True


def insertVectorDataset(dbms: mg.mongoConnection.startSession, datasetName: str, fileName: str, metadata: dict):
    """
        Insereix els continguts del fitxer de dades a la BD

        :param dbms: connexió a la base de dades.
        :param datasetName: nom del conjunt de dades.
        :param fileName: nom del fitxer (ruto completa) que contés les dades.
        :param metadata: Diccionari amb les metadades del conjunt de dades.

        :return: True si la importació s'ha fet correctament.
    """

    database = dbms.bd  # Connects to Outliers DB

    logging.warning("Creating collection {}".format(datasetName))
    collection = database.create_collection(datasetName)  # Collection creation

    with open(fileName, 'r') as file:

        for vector in file.readlines():
            vector = vector.strip('\n')
            fields = vector.split(',')

            if 'id' in metadata:  # Dataset has an identification field

                intel = [float(fields[x]) for x in range(len(fields)) if
                         x != metadata['label_pos'] and x != metadata['id']]

                label = metadata['labels'][str(fields[metadata['label_pos']])]

                collection.insert_one(
                    {"id": fields[metadata['id']], "label": label, "vector": intel, "utility": "Outliers"})

            else:
                intel = [float(fields[x]) for x in range(len(fields)) if x != metadata['label_pos']]
                if len(fields) > 0:
                    collection.insert_one(
                        {"label": fields[metadata['label_pos']], "vector": intel, "utility": "Outliers"})

    return True


def insertImageDataset(dbms: mg.mongoConnection.startSession, datasetName: str, folderName: str, metadata: dict):
    """
    S’encarrega d’inserir a la BD tota la informació dels datasets d'imatges.

    :param dbms: connexió a la base de dades.
    :param datasetName: nom del conjunt de dades.
    :param folderName: nom de la carpeta (ruto completa) que contés les dades.
    :param metadata: Diccionari amb les metadades del conjunt de dades.

    :return: True si la importació s'ha fet correctament.
    """

    database = dbms.bd  # Connects to Outliers DB
    logging.warning("Creating collection {}".format(datasetName))
    collection = database.create_collection(datasetName)  # Collection creation

    label_path = folderName + metadata['label_folder']
    label_dict = dict()

    files = os.listdir(label_path)

    for file in files:  # Labels for all photos with standarized categories
        thing = str(file).split('.')

        with open(label_path + '/' + file, 'r') as label:
            for lab in label.readlines():

                if int(lab) not in label_dict.keys():
                    label_dict[int(lab)] = [thing[0]]

                else:
                    label_dict[int(lab)].append(thing[0])

    descriptor_dict, other_features = insertDescriptors(folderName, metadata)

    for img in label_dict:
        p = 'images' + '/im' + str(img) + '.jpg'  # Path to image

        if img in descriptor_dict:
            data = {x[1]: x[0] for x in descriptor_dict[img]}
            for other in other_features:
                data[other[1]] = other[0]

            collection.insert_one(
                {"_id": img, "label": label_dict[img], "path": p, "descriptors": data, "utility": "Outliers"})

        else:
            collection.insert_one(
                {"_id": img, "label": label_dict[img], "path": p, "utility": "Outliers"})

    logging.warning("Inserted info image dataset {!r}".format(datasetName))

    return True


def insertDescriptors(fileName: str, metadata: dict):
    """
    S’encarrega d’inserir a la BD tota la informació de les característiques que s’han extret de les imatges.

    :param dataset: nom del conjunt de dades.
    :param fileName: nom del fitxer (ruto completa) que contés les dades.
    :param metadata: Diccionari amb les metadades del conjunt de dades.

    :return:  NO.
    """

    # TODO: Implementa aquesta funció tenim en compte les metadades. El fitxer de caracteristiques es binari. Mireu
    #  la documentació per veure la seva estructura.

    descriptor_dict = {}
    other_features = []

    path = fileName + metadata['descriptor_folder']  # Descriptors path

    descriptors = os.listdir(path)  # All aviable descriptors

    for descriptor in descriptors:
        path_to = path + '/' + descriptor
        descriptor_name = descriptor.split('.')

        if descriptor_name[-1] == 'Labelfeatures':
            data = open(path_to, 'rb').readlines()
            form = data[0].decode('utf-8').split(' ')
            buf = int(form[0])  # Total matrix size
            vec = int(form[1])  # Elements for vector

            signal = list()  # Matrix-like list for later on insertion

            for i in range(buf):  # Binary file reading
                row = list()
                for j in range(vec):
                    value = struct.unpack_from('f', data[1], 4 * (24 * i + j))
                    row.append(value[0])

                signal.append(row)

            for i in range(len(signal)):
                if i not in descriptor_dict:
                    descriptor_dict[i] = [(signal[i], descriptor.replace('.', '_'))]

                else:
                    descriptor_dict[i].append((signal[i], descriptor.replace('.', '_')))

        elif descriptor_name[-1] == 'Sigmoidfeatures' or descriptor_name[-1] == 'Visualfeatures':  # Same file format
            recon = True
            data = open(path_to, 'rb').readlines()

            signal = [x for x in data[1]]  # These files already are ints (IDK why)

            other_features.append((signal, descriptor.replace('.', '_')))

    # logging.warning("Features of dataset {} correctly inserted".format(dataset))
    return descriptor_dict, other_features


def existsDataset(dbms: mg.mongoConnection.startSession, dataset: str, filename: str):
    """
    Comprova que el nombre de registres (documents) inserits de cada dataset coincideixi amb el nombre de registres
    (files) en el fitxer de dades  utilitzat per la importació de les dades. Si els dos valors coincideixen retorna
    cert, i fals en cas contrari.

    :param dbms: connexió a la base de dades
    :param dataset: nom del datase
    :param filename: nom del fitxer

    :return: True si tots els vectors de característiques estan inserits
    """

    if dataset in dbms.bd.list_collection_names():

        docs_count = dbms.bd[dataset].count_documents({})  # Count all docs

        with open(filename, 'r') as file:  # Row count
            file_count = len(file.readlines())

        if docs_count == file_count:
            return True

        else:
            print("Elements haven't been inserted properly; Cleaning")
            if deleteDataset(dbms, dataset):
                logging.warning("Cleaned")
                return False

        logging.warning("ERROR: Elements haven't been deleted correctly")

    return False


def insert(dbms, datasetName, fileName, metadata=None, type='vector'):
    """
    Insert the data set provided in the file called fileName

    Input arguments:
    :param dbms: object with the data connection
    :param datasetName: Name of the dataset
    :param fileName: full path and name of the file containing the data to be inserted
    :param matadata: dataset specific parameters required to properly insert the data.
    :param type: vector or image

    :return: None
    """

    if type == "vector":
        if existsDataset(dbms, datasetName, fileName):
            logging.warning("Dataset {} already inserted".format(datasetName))
            return True

        # Afegim la info en la taula general del dataset
        return insertVectorDataset(dbms, datasetName, fileName, metadata=metadata)

    if type == "image":
        name = fileName.split(os.sep)

        if datasetName in dbms.bd.list_collection_names():  # Special way to check images counting docs and pics
            if dbms.bd[datasetName].count_documents({}) == 24581:
                logging.warning("Image dataset {} already inserted".format(datasetName))
                return True
        else:
            """
            Inserim la informació relacionada amb la base de dades de la mirflickr (nom de les imatges i anotacions a 
            la BD
            """
            return insertImageDataset(dbms, datasetName, fileName, metadata=metadata)


if __name__ == '__main__':
    # read commandline arguments, first
    # fullCmdArguments = sys.argv

    # Parse options
    opts = Options()
    args = opts.parse()

    featuresFile = None
    db = None
    isConfigFile = False
    data = None
    insertData = True

    if args.config is not None:
        with open(args.config) as f:
            data = json.load(f)
    else:
        opts.print_help()
        sys.exit(1)

    db = mg.mongoConnection(data)

    if db is None:
        opts.print_help()
        sys.exit(1)

    """Iniciem la sessio"""
    db.startSession()

    if db.testConnection():
        logging.warning("La connexió a MongoDB funciona correctament.")

    if args.fileName is None:
        logging.warning("Falta el nom del fitxer de dades ")
        opts.print_help()
        db.close()
        sys.exit(-1)

    if args.metadata is not None:
        with open(args.metadata) as f:
            metadata = json.load(f)
        metadata = metadata[args.datasetName.lower()]
    else:
        metadata = dict()

    # inserim les dades
    res = insert(db, args.datasetName, args.fileName, metadata=metadata, type=args.dataType)

    if res:
        logging.warning("Dades carregades correctament")
    else:
        logging.warning("Problemes al inserir les dades")

    db.close()
    sys.exit(0)
