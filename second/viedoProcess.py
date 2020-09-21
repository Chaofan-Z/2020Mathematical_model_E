import cv2


def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)

videoCapture = cv2.VideoCapture("Fog.mp4")
sucess = True
i, j, timeF = 0, 1, 1500
while sucess:
    if i < 850:
        sucess, frame = videoCapture.read()
        i += 1
        continue
    sucess, frame = videoCapture.read()
    if (i-850) % timeF == 0:
        save_image(frame, './output/image/', j)
        j += 1
        print(j)
    i += 1
