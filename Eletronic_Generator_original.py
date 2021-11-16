import os, random
import cv2, argparse
import numpy as np

def random_bright(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        # Plate
        self.plate = cv2.imread("plate_b.jpg")

        # loading Number
        file_path = "./num/"
        file_list = os.listdir(file_path)
        self.Number = list()
        self.Number_thr = list()
        self.Number_thr1 = list()
        self.number_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            # thr plate
            _, thr = cv2.threshold(img, 127 , 1 ,cv2.THRESH_BINARY)
            # thr img
            _, thr1 = cv2.threshold(img, 127 , 1 ,cv2.THRESH_BINARY_INV)
            self.Number.append(img)
            self.Number_thr.append(thr)
            self.Number_thr1.append(thr1)
            self.number_list.append(file[0:-4])

        # loading Char
        file_path = "./char1/"
        file_list = os.listdir(file_path)
        self.Char1 = list()
        self.Char1_thr = list()
        self.Char1_thr1 = list()
        self.char_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            # thr plate
            _, thr = cv2.threshold(img, 127 , 1 ,cv2.THRESH_BINARY)
            # thr img
            _, thr1 = cv2.threshold(img, 127 , 1 ,cv2.THRESH_BINARY_INV)
            self.Char1.append(img)
            self.Char1_thr.append(thr)
            self.Char1_thr1.append(thr1)
            self.char_list.append(file[0:-4])

    def Type_6(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        number_thr = [cv2.resize(number_thr, (56, 83)) for number_thr in self.Number_thr]
        number_thr1 = [cv2.resize(number_thr1, (56, 83)) for number_thr1 in self.Number_thr1]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        char_thr = [cv2.resize(char1_thr, (60, 83)) for char1_thr in self.Char1_thr]
        char_thr1 = [cv2.resize(char1_thr1, (60, 83)) for char1_thr1 in self.Char1_thr1]
        Plate = cv2.resize(self.plate, (520, 110))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (520, 110))
            label = "Z"
            # row -> y , col -> x
            row, col = 13, 44  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = Plate[row:row + 83, col:col + 56, :]* number_thr[rand_int]  + number[rand_int] * number_thr1[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = Plate[row:row + 83, col:col + 56, :]* number_thr[rand_int]  + number[rand_int] * number_thr1[rand_int]
            col += 56

            # character 3
            label += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = Plate[row:row + 83, col:col + 60, :]* char_thr[i%37] + char[i%37] *  char_thr1[i%37]
            col += (60 + 36)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = Plate[row:row + 83, col:col + 56, :]* number_thr[rand_int]  + number[rand_int] * number_thr1[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] =  Plate[row:row + 83, col:col + 56, :]* number_thr[rand_int]  + number[rand_int] * number_thr1[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = Plate[row:row + 83, col:col + 56, :]* number_thr[rand_int]  + number[rand_int] * number_thr1[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = Plate[row:row + 83, col:col + 56, :]* number_thr[rand_int]  + number[rand_int] * number_thr1[rand_int]
            col += 56
            Plate = random_bright(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="../CRNN/DB/")
parser.add_argument("-n", "--num", help="number of image",
                    type=int)
parser.add_argument("-s", "--save", help="save or imshow",
                    type=bool, default=True)
args = parser.parse_args()


img_dir = args.img_dir
A = ImageGenerator(img_dir)

num_img = args.num
Save = args.save

A.Type_6(num_img, save=Save)
print("Type 6 finish")