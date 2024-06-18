import cv2
import numpy as np

# Carregar o modelo YOLOv3 e as classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Carrega o modelo YOLOv3 pré-treinado
layer_names = net.getLayerNames()  # Obtém os nomes das camadas do modelo
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Obtém as camadas de saída do modelo

# Carrega as classes para identificação de objetos
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]  # Lê e armazena as classes a partir do arquivo 'coco.names'

# Função para capturar vídeo da webcam e detectar objetos
def capture_and_detect():
    cap = cv2.VideoCapture(0)  # Inicia a captura de vídeo da webcam
    while cap.isOpened():
        ret, frame = cap.read()  # Lê um quadro do vídeo
        if not ret:
            break  # Se não houver quadro, sai do loop

        # Redimensiona a imagem para 416x416 para a YOLO e cria um blob
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)  # Define a entrada da rede neural YOLO como o blob
        outs = net.forward(output_layers)  # Realiza a detecção de objetos e obtém as saídas

        # Processa as saídas da YOLO para obter caixas delimitadoras, confianças e IDs de classes
        class_ids = []
        confidences = []
        boxes = []
        height, width, channels = frame.shape
        for out in outs:
            for detection in out:
                scores = detection[5:]  # Scores de confiança para cada classe
                class_id = np.argmax(scores)  # Obtém o ID da classe com maior confiança
                confidence = scores[class_id]  # Obtém a confiança da classe detectada
                if confidence > 0.5:  # Considera apenas detecções com confiança maior que 0.5
                    # Calcula as coordenadas da caixa delimitadora
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Aplica Non-Maximum Suppression (NMS) para suprimir caixas delimitadoras redundantes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Desenha caixas delimitadoras e rótulos na imagem
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])  # Obtém o nome da classe usando o ID
                color = (0, 255, 0)  # Cor da caixa delimitadora e do rótulo
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # Desenha a caixa delimitadora
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Adiciona o rótulo

        # Mostra a imagem capturada com as caixas delimitadoras e rótulos
        cv2.imshow('Webcam', frame)

        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Libera o objeto de captura de vídeo
    cv2.destroyAllWindows()  # Fecha todas as janelas abertas pelo OpenCV

# Chama a função para iniciar a detecção de objetos na webcam
capture_and_detect()
