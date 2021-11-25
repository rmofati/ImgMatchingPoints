# Resolução do Exercício 4 - FAST e SSD
# Aluno: Rafael Mofati Campos

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("imLivro1R.jpg", 0)
img2 = cv2.imread("imLivro3R.jpg", 0)

# Desenvolvimento do exercício 1
fast = cv2.FastFeatureDetector_create(65, True, 2)
kp = fast.detect(img1,None)
img1_result = cv2.drawKeypoints(img1, kp, None, color=(0,0,255))

print("+--------- Resultados de Pontos - Imagem 1 ---------+")
print( "Número de pontos encontrados: {}".format(len(kp)))

pts1 = cv2.KeyPoint_convert(kp)
pts1_x = [int(pts1[i][0]) for i in range(len(pts1))]
pts1_y = [int(pts1[i][1]) for i in range(len(pts1))]
pts1 = [list(pair) for pair in zip(pts1_x, pts1_y)]

print(pts1)
print()

cv2.imwrite('img1_result.png',img1_result)


kp = fast.detect(img2,None)
img2_result = cv2.drawKeypoints(img2, kp, None, color=(0,0,255))

print("+--------- Resultados de Pontos - Imagem 2 ---------+")
print( "Número de pontos encontrados: {}".format(len(kp)))

pts2 = cv2.KeyPoint_convert(kp)
pts2_x = [int(pts2[i][0]) for i in range(len(pts2))]
pts2_y = [int(pts2[i][1]) for i in range(len(pts2))]
pts2 = [list(pair) for pair in zip(pts2_x, pts2_y)]

print(pts2)
print()

cv2.imwrite('img2_result.png',img2_result)

# Apresentação de resultados do exercício 1
plt.suptitle('RESULTADOS')
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.subplot(2, 2, 2)
plt.imshow(img1_result)
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.subplot(2, 2, 4)
plt.imshow(img2_result)
plt.show()

# Desenvolvimento do exercício 2
def patch(img, x, y, patch_size=19):
    x1 = x - int(patch_size/2)
    x2 = x + int(patch_size/2)
    y1 = y - int(patch_size/2)
    y2 = y + int(patch_size/2)
    patch = img[y1:y2,x1:x2]
    return patch

def calc_ssd(patch1, patch2):
    ssd = np.sum((patch1-patch2)**2)
    return ssd

dict_points = {}

for i in range(len(pts1)):
    img1_patch = patch(img1, pts1[i][0], pts1[i][1])
    for j in range(len(pts2)):
        img2_patch = patch(img2, pts2[j][0], pts2[j][1])
        dict_points[(i,j)] = calc_ssd(img1_patch, img2_patch)

list_min_values = []
list_min_point = []

for i in range(len(pts1)):
    min_value = 999999
    list_min_point.append(dict_points[(i,0)])
    for j in range(len(pts2)):
        if dict_points[(i,j)] < min_value:
            min_value = dict_points[(i,j)]
            list_min_point[i] = (i,j)
    list_min_values.append(min_value)

print(list_min_values)
print(list_min_point)

print()

# Teste de correspondência manual
list_resposta = []

for k in range(len(list_min_point)):
    print("+===============================================================================+")
    print("Correspondência (Ponto Img1, Ponto Img2):",list_min_point[k])
    print("Coordenadas de cada imagem [Img1]-[Img2]:",pts1[list_min_point[k][0]],"-",pts2[list_min_point[k][1]])
    print()

    window1_name = "Imagem 1"
    window2_name = "Imagem 2"
    coordinates1 = (pts1[list_min_point[k][0]][0], pts1[list_min_point[k][0]][1])
    coordinates2 = (pts2[list_min_point[k][1]][0], pts2[list_min_point[k][1]][1])
    radius = 3
    color = (0,255,0)
    thickness = 2

    img1 = cv2.imread("imLivro1R.jpg", 0)
    img2 = cv2.imread("imLivro3R.jpg", 0)

    image1 = cv2.circle(img1, coordinates1, radius, color, thickness)
    image2 = cv2.circle(img2, coordinates2, radius, color, thickness)

    scale = 3
    width = int(image1.shape[1] * scale)
    height = int(image2.shape[0] * scale)
    dim = (width, height)

    image1_resized = cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)
    image2_resized = cv2.resize(image2, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow(window1_name, image1_resized)
    cv2.imshow(window2_name, image2_resized)

    print("=> Aperte (E) para correspondência ERRADA ou (C) para correspondência CORRETA:")

    key = cv2.waitKey(0)
    if key == 69 or key == 101: #Errado
        list_resposta.append("E")
    elif key == 99 or key == 67: #Certo
        list_resposta.append("C")

    cv2.destroyAllWindows()

print()
print("+======================================== RESULTADO FINAL =========================================+")
print("Resultados:",list_resposta)
print("Total de pontos:",len(list_min_point)," | Total de respostas: ",len(list_resposta))
print("Total de respostas erradas:",list_resposta.count("E")+list_resposta.count("e"))
print("Total de respostas corretas:",list_resposta.count("C")+list_resposta.count("c"))
print("+======================================== RESULTADO FINAL =========================================+")


#+======================================== RESULTADO FINAL =========================================+
#Todos os resultados: ['E','E','E','C','E','E','C','C','E','C','E','E','C','E','E','E','E','C','E','E','C','C','E','E','E','E','C','E','E','E','E','E','C','E','C','E','E','C','E','E','E','E','E','C','E','E','C','E','E','E','E','E','E','E','C','E','C','C']
#Total de pontos: 58  | Total de respostas:  58
#Total de respostas erradas: 41 (70,7%)
#Total de respostas corretas: 17 (29,3%)