# Preprocess training images.
# Scale 300 seems to be sufficient; 500 and 1000 may be overkill
import cv2
import glob
import numpy
import os
import settings

logger = settings.getlogger("preprocess_images")


def scaleRadius(img, scale):
    x = img[img.shape[0]//2, :, :].sum(1)
    r = (x > x.mean()/10).sum()/2
    # print("r", r)
    s = scale*1.0/r
    # print("s", s)   # 0-1 float
    resized_img = cv2.resize(img, (0, 0), fx=s, fy=s)
    # print("resized_img", resized_img)

    return resized_img


def preprocessImages(scale):
    # for scale in [300, 500, 1000]:
    # for f in (glob.glob(settings.DR_SRC_DIR + "train/*.jpeg") + glob.glob(settings.DR_SRC_DIR + "test/*.jpeg")):
    # for f in (glob.glob(settings.WORKING_DIR + "dataset/train/*/*.jpeg") + glob.glob(settings.WORKING_DIR + "test/*/*/*.jpeg")):
    for f in (glob.glob(settings.WORKING_DIR + "dataset/train/*/*.jpeg")):
        try:
            a = cv2.imread(f)
            # print("cv2.imread(f):", a)
            # scale img to a given radius
            # rescale the images to have the same radius (300 pixels or 500 pixels)
            a = scaleRadius(a, scale)

            # remove outer 10%. clip the images to 90% size to remove the boundary effects.
            b = numpy.zeros(a.shape)
            cv2.circle(b, (a.shape[1]//2, a.shape[0]//2), int(scale*0.9), (1, 1, 1), -1, 8, 0)

            # subtract local mean color.
            # subtract the local average color; the local average gets mapped to 50% gray,
            aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale/30), -4, 128)*b+128*(1-b)
            # print("aa", aa)  # [[128. 128. 128.][128. 128. 128.][128. 128. 128.]...[128. 128. 128.]]

            # result_img_dir = str(scale) + "_" + f
            # fpath, fname = os.path.split(result_img_dir)
            # if not os.path.exists(fpath):
            #     os.makedirs(fpath)

            parts = f.split("/")
            result_img_dir = settings.WORKING_DIR + "dataset/" + str(scale) + "_train/" + parts[-2] + "/" + parts[-1]
            logger.info("result_img_dir:{0}".format(result_img_dir))
            fpath, fname = os.path.split(result_img_dir)
            if not os.path.exists(fpath):
                os.makedirs(fpath)

            # aa = cv2.resize(aa,(512,512))

            cv2.imwrite(result_img_dir, aa)
        except:
            logger.info("{0}: went wrong".format(f))


if __name__ == '__main__':
    logger.info("Preprocessing the train data...")
    SCALE = 300
    preprocessImages(scale=SCALE)
