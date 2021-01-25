import numpy as np
import imageio as img
from time import sleep
import sounddevice as sd
from scipy.io import wavfile
from scipy.ndimage import convolve
from matplotlib import pyplot as plt

import cv2 as cv


class Sound:
    SampleRate = 0

    @staticmethod
    def open(path: str):
        """
        :param path: path to the target wave file
        :return: audio signal in np.ndarray form
        """
        Sound.SampleRate, sig = wavfile.read(path)

        return sig

    @staticmethod
    def draw(sig: np.ndarray, length: float, sig2: np.ndarray = None):
        """
        :param sig: the target signal to be drawn
        :param length: the length of the signal you want to be plotted
        :param sig2: [optional] second target signal to be drawn
        :return: None
        """
        length = int(length * Sound.SampleRate)
        length = min(len(sig), length) if sig2 is None else min(len(sig), len(sig2), length)
        if sig2 is not None:
            plt.subplot(2, 1, 1)
            plt.plot(sig2[:length])
            plt.subplot(2, 1, 2)
        else:
            plt.subplot(1, 1, 1)
        plt.plot(sig[:length])
        plt.show()

    @staticmethod
    def play(sig: np.ndarray):
        sd.play(sig, Sound.SampleRate)
        sleep(len(sig) / Sound.SampleRate)

    @staticmethod
    def amplify(sig: np.ndarray, amp: float):

        return (amp * sig).astype(np.int16)

    @staticmethod
    def delay(sig: np.ndarray, t: float):
        n0 = t * Sound.SampleRate
        if t > 0:
            new_sig = np.copy(sig[n0: ])
        else:
            new_sig = np.copy(sig[: -n0])

        return new_sig

    @staticmethod
    def noise(sig: np.ndarray, gain: float):
        """
        در هر لحظه، مقدار سیگنال با ضریبی از یک مقدار تصادفی نرمال جمع می شود. به طور مثال، از صفر تا 1000، مقدار سیگنال اصلی
        صفر می باشد، ولی با فراخوانی تابع نویز، سیگنال از 0 تا 1000 نیز مقدار مخالف صفر به خود می گیرد، که چون ضریبی از
        مقدار تصادفی که با احتمال برابر منفی یا مثبت می باشد، این مقادیر هم با احتمال برابر تقریباً gain- و gain+ می شوند.
        در نهایت نیز با پخش دوباره فایل، فهمیدیم که صدا نویز دارد :)

        """
        noise = np.random.normal(0, 1, len(sig))

        noise = noise * gain
        noised_sig = sig + noise
        return noised_sig

    @staticmethod
    def echo(sig: np.ndarray, dt: float, echo_factor: float):
        """
         در هر لحظه، سیگنال، با ضریبی دو مقدار قبلی نیز خود نیز جمع می شود. در واقع در هر لحظه، سیگنال 3 لحظه شنیده می شود و به نوعی اکو حاصل می شود.



        """
        n0 = int(dt * Sound.SampleRate)
        echoed_sig = np.copy(sig)

        for i in range(len(sig)):
            if i - n0 > 0:

                echoed_sig[i] += echo_factor * sig[i - n0]
            elif i - 2 * n0 > 0:
                echoed_sig[i] += echo_factor * sig[i - n0]
                echoed_sig[i] += echo_factor**2 * sig[i - 2 * n0]
            elif i - 3 * n0 > 0:
                echoed_sig[i] += echo_factor * sig[i - n0]
                echoed_sig[i] += echo_factor ** 2 * sig[i - 2 * n0]
                echoed_sig[i] += echo_factor ** 3 * sig[i - 3 * n0]
        return echoed_sig

    @staticmethod
    def speed(sig: np.ndarray, spd: float):

        if spd > 1:

            num_of_removals = len(sig) - int(len(sig) / spd)
            step = int(len(sig) / num_of_removals)

            to_remove = []
            index = 0
            for i in range(num_of_removals):
                to_remove.append(index)
                index = int(index + step)

            return np.delete(sig, to_remove)

        elif spd < 1:

            num_of_appendants =  int(len(sig) / spd) - len(sig)
            step = len(sig) // num_of_appendants
            index = 0
            to_append = np.ones(len(sig)).astype(np.int16)


            for i in range(num_of_appendants):
                to_append[int(index)] += 1
                index += step
            return np.repeat(sig, to_append)
        else:
            return sig

class Image:
    @staticmethod
    def open(path: str):
        pic = img.imread(path)
        return pic

    @staticmethod
    def convert_to_rgb(pic_sig: np.ndarray):
        red = pic_sig[:, :, 0]
        green = pic_sig[:, :, 1]
        blue = pic_sig[:, :, 2]
        return red, green, blue

    @staticmethod
    def show(im: np.ndarray):
        plt.imshow(im)
        plt.show()

    @staticmethod
    def rgb_show(r: np.ndarray, g: np.ndarray, b: np.ndarray, ):
        plt.imshow(np.dstack([r, g, b]))
        plt.show()

    @staticmethod
    def gray_show(gray: np.ndarray):
        plt.imshow(np.dstack([gray, gray, gray]))
        plt.show()

    @staticmethod
    def convolve(pic: np.ndarray, fil: np.ndarray):
        #return np.array(np.minimum(convolve(pic, fil), 255), dtype='uint8')

        return cv.filter2D(pic, -1, fil)
    @staticmethod
    def brightness(pic: np.ndarray, b: float):

        pic = pic * b

        pic = pic.astype( dtype='uint8')



        #pic[pic > 255] = 255
        return pic

    @staticmethod
    def gray_scale(pic: np.ndarray):
        shape = np.shape(pic)

        grey = np.array([ [0 for i in range(shape[1])] for j in range(shape[0] )])

        for i in range(shape[0]):
            for j in range(shape[1]):
                grey[i][j] = pic[i][j][0] * 0.2126 +  pic[i][j][1] * 0.7152 + pic[i][j][2] * 0.0722
        return grey

    @staticmethod
    def black_and_white(gray: np.ndarray, border: int):
        gray[gray > border] = 255
        gray[gray <= border] = 0
        return gray

    @staticmethod
    def first_filter(pic):
        """

        تصویر را تار می کند.
        """
        kernel = np.array([[1 / 9 for i in range(3)] for j in range(3)])

        filtered_turing = Image.convolve(pic, kernel)
        return filtered_turing

    @staticmethod
    def second_filter(pic):
        """
        قسمت هایی از عکس را که تفاوت میان آن پیکسل با پیکسل های مجاورش زیاد است را پر رنگ تر نشان می دهد. در واقع
        تفاوت ها بیشتر نشان داده می شوند.
        """
        kernel = np.array([[-1/9, -1/9, -1/9], [-1/9, 8/9, -1/9], [-1/9, -1/9, -1/9]]) * 9

        return Image.convolve(pic, kernel)



    @staticmethod
    def third_filter(pic, alpha):

        flt = np.array([[alpha / 9 for i in range(3)] for j in range(3)])
        flt = flt + np.array([[0, 0, 0], [0, 1 - alpha, 0], [0, 0, 0]])

        filtered_turing = Image.convolve(pic, flt)
        return filtered_turing

    @staticmethod
    def filter1(pic):
        """

        calculates horizontal edges
        """

        kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) * 1 / 8
        return Image.convolve(pic, kernel)

    @staticmethod
    def filter2(pic):
        """

        calculates vertical edges
        """
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * 1/8
        return Image.convolve(pic, kernel)

if __name__ == '__main__':
    # your codes here
    sig = Sound.open('./intro.wav')
    #Sound.play(sig)
    #Sound.draw(sig, 1)

    #amp_sig = Sound.amplify(sig, 0.1)
    #Sound.play(amp_sig)

    #delayed_sig = Sound.delay(sig, -2)
    #Sound.play(delayed_sig)

    #noised_sig = Sound.noise(sig, 1000)
    #Sound.play(noised_sig)
    #Sound.draw(sig, 1, noised_sig)

    #echoed_sig = Sound.echo(sig, 0.3, 0.5)
    #Sound.play(echoed_sig)
    #Sound.draw(sig, 1, echoed_sig)


    #speeded_sig = Sound.speed(sig, 0.8)
    #Sound.play(speeded_sig)



    azadi_img = Image.open("./Azadi.jpg")
    #Image.show(azadi_img)
    #converted_to_rgb = Image.convert_to_rgb(azadi_img)

    #Image.rgb_show(*converted_to_rgb)


    #brightened_img = Image.brightness(azadi_img,0.5)

    #Image.show(brightened_img)


    #grey_scaled = Image.gray_scale(azadi_img)
    #Image.gray_show(grey_scaled)


    #bw_img = Image.black_and_white(grey_scaled, 200)
    #Image.gray_show(bw_img)

    turing_img = Image.open('./turing.jpg')
    #filtered_img = Image.first_filter(turing_img)
    #Image.show(filtered_img)


    big_azadi = Image.open('./Azadi_big.jpg')
    #filtered_img = Image.second_filter(big_azadi)
    #Image.show(filtered_img)

    oppenheim_img = Image.open('./oppenheim.jpg')
    filtered_img = Image.third_filter(oppenheim_img, 10)

    Image.show(filtered_img)


    wall_image = Image.open('./wall.jpg')
    #Image.show(wall_image)

    #first_filtered = Image.filter1(wall_image)
    #Image.show(first_filtered)

    #second_filtered = Image.filter2(wall_image)

    #Image.show(second_filtered)

    """
     both horizontal and vertical edges are seen
    """
    #Image.show(first_filtered + second_filtered)

