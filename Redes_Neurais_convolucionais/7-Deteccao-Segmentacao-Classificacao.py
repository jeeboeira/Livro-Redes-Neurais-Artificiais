import numpy  as np
import pandas as pd
import os
import json
import shutil
import random
import cv2
import torch
import torchvision
import detectron2


from PIL                         import Image                               # Lê e carrega imagens
from pathlib                     import Path                                # Resolve endereços de carregamento dos arquivos
from pycocotools                 import mask as cocomask                    # importa Mask pois temos as imagens e as máscaras
from matplotlib                  import pyplot as plt 
from detectron2                  import model_zoo                           # Arquitetura de redes neurais pré moldadas
from detectron2.config           import get_cfg                             # define configurações específicas de modelo pré moldado
from detectron2.data             import DatasetCatalog 
from detectron2.data             import MetadataCatalog                     # Permite carregar metadados (caracteristicas e configurações) de nossa base
from detectron2.data             import build_detection_train_loader        # Faz o treino da rede
from detectron2.data             import build_detection_test_loader         # Faz o teste da rede
from detectron2.data.datasets    import register_coco_instances             # Após o tratamento, mapeia e registra como instancia para processamento do modelo COCO
from detectron2.engine           import DefaultTrainer                      # aplica as métricas de avaliação do treino
from detectron2.engine           import DefaultPredictor                    # aplica as métricas de avaliação dos testes
from detectron2.modeling         import build_model                         # Constrói e salva o modelo
from detectron2.solver           import build_lr_scheduler, build_optimizer # Avaliação da taxa de aprendizado
from detectron2.utils.visualizer import Visualizer                          # Permite visualizar caracteristicas da amostra processada

# Referência da pasta
pastaArquivos = r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\7Sete\Brain_Tumor_Detection'
# Verifica os dados
dados_imagens = pd.read_csv(pastaArquivos + '\data.csv')
#print(dados_imagens.head())
#print(dados_imagens.shape)

# Mapa que retorna em lista ordenada o caminho pra cada imagem na base
mapa = sorted(list(Path(pastaArquivos).rglob('*tif')))
#print(mapa[0:5])
#print(len(mapa))

# Informações pertinentes de projeto
info = {"year"         : 2024,
        "version"      : "1.0",
        "description"  : "Segmentação de Tumores Cerebrais",
        "contributor"  : "Jessé Boeira",
        "url"          : "https://github.com/jeeboeira",
        "data_created" : "2024"}
licenses = [{"id"   : 1,
             "name" : "Attribution-NonCommercial",
             "url"  : "http://creativecommons.org/licenses/by-nc-sa/2.0/"}]
type = "instances"

# Arrays onde serão guardadas as características de cada imagem
masks       = []
names       = []
bboxes      = []
areas       = []
annotations = []
images      = []
id          = 0

# Verifica o numero de caracteres no caminho das imagens
#print(len(str(pastaArquivos + '\TCGA_CS_4941_19960909\TCGA_CS_4941_19960909_10_mask.tif')))


###################################
# Início do Tratamento de Imagens #
###################################

# Cria um contorno quadrado onde quero identificar, e depois contorna a imagem em si
for im in mapa:
    if len(str(im)) == 193 or len(str(im)) == 194 :
        msk = np.array(Image.open(im).convert('L'))   # Converte a imagem em um array
        contours, _ = cv2.findContours(msk,           # Cria uma variável contours e uma variável nula de nome
                                       cv2.RETR_TREE, # Filtros de detecção de saliência
                                       cv2.CHAIN_APPROX_SIMPLE)
        segmentation = [] # Recebe os dados de segmentação quando houver, nem toda imagem tem uma massa tumoral detectada e segmentada
        crowd        = 0  # Salva o número de segmentações realizadas

        for contour in contours:
            if contour.size >= 6: # Filtra contorno maior que 6, pois menor pode ser artefatos ou ruídos
                crowd += 1
                segmentation.append(contour.flatten().tolist())

        if crowd > 1:
            iscrowd = 1
        else:
            iscrowd = 0
        try:
            RLEs = cocomask.frPyObjects(segmentation,
                                        msk.shape[0],
                                        msk.shape[1])
            RLE  = cocomask.merge(RLEs)
            area = float(cocomask.area(RLE))
        except:
            area = []

        [x, y, w, h] = cv2.boundingRect(msk) # Cria Bounding Box
        bbox1 = [float(x), float(y), float(w), float(h)]
        areas.append(area)
        id += 1

        # Variável que tentará ler o tamanho da variável segmentation
        try:
            for s in range(len(segmentation[0]) - 1):
                segmentation[0][s] = float(segmentation[0][s])
        except:
            pass

        annotations.append({"segmentation" : segmentation,
                            "area"         : area,
                            "iscrowd"      : iscrowd,
                            "image_id"     : id,
                            "bbox"         : bbox1,
                            "category_id"  : 1,
                            "id"           : id})
        
        images.append({"date_captured" : "2024",
                       "file_name"     : str(im)[:-9]+".tif",
                       "id"            : id,
                       "license"       : 1,
                       "url"           : "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                       "height"        : msk.shape[0],
                       "width"         : msk.shape[1]})
#print(annotations[0])
#print(len(annotations))
#print(annotations[46])
#print(images[0])
#print(len(images))

# Dict confirmando a presença de tumor
categoria_seg = [{'id'            : 1,
                  'name'          : 'tumor',
                  'supercategory' : 'shape'}]
#print(categoria_seg)

# Atrela todos os dados pertinentes a cada imagem
coco_output = {"info"        : info,
               "licenses"    : licenses,
               "categories"  : categoria_seg,
               "images"      : [],
               "annotations" : []}
#print(coco_output)


# Geração das fichas técnicas de cada imagem
for im_id in images:
    coco_output["images"].append(im_id)
for annotation_id in annotations:
    coco_output["annotations"].append(annotation_id)
with open(pastaArquivos+'/annotations_imagens.json', 'w') as output_json_file:
    json.dump(coco_output, output_json_file)

shutil.copy (pastaArquivos + '/annotations_imagens.json',
             pastaArquivos + '/projeto/annotations_imagens.json')

register_coco_instances("treino", {},
                        pastaArquivos +  '/annotations_imagens.json',
                        pastaArquivos)
register_coco_instances("validation", {},
                        pastaArquivos +  '/annotations_imagens.json',
                        pastaArquivos)

#Fim polimento de dados

###################
# Data Processing #
###################

treino_metadata     = MetadataCatalog.get("treino")
validation_metadata = MetadataCatalog.get("validation")

#print(treino_metadata)
#print(validation_metadata)

treino_dict     = DatasetCatalog.get("treino")
validation_dict = DatasetCatalog.get("validation")

#print(treino_dict[0])

for item in random.sample(treino_dict, 3):
    print(item)
    imagem_nome = cv2.imread(item["file_name"])
    visualizer = Visualizer(imagem_nome[:, :, ::-1],
                            metadata = MetadataCatalog.get("treino"),
                            scale=2)
    vis = visualizer.draw_dataset_dict(item)
    imagem_treino = vis.get_image()[:, :, ::-1]
    plt.figure(figsize= (10,6))
    plt.imshow(imagem_treino)