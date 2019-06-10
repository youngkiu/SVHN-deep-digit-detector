#-*- coding: utf-8 -*-
import cv2
import numpy as np
import keras.models

import digit_detector.region_proposal as rp
import digit_detector.show as show
import digit_detector.detect as detector
import digit_detector.file_io as file_io
import digit_detector.preprocess as preproc
import digit_detector.classify as cls

detect_model = "detector_model.hdf5"
recognize_model = "recognize_model.hdf5"

mean_value_for_detector = 107.524
mean_value_for_recognizer = 112.833

model_input_shape = (32,32,1)
DIR = '../datasets/svhn/train'

if __name__ == "__main__":
    # 1. image files
    img_files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=None, random_order=False)

    preproc_for_detector = preproc.GrayImgPreprocessor(mean_value_for_detector)
    preproc_for_recognizer = preproc.GrayImgPreprocessor(mean_value_for_recognizer)

    char_detector = cls.CnnClassifier(detect_model, preproc_for_detector, model_input_shape)
    char_recognizer = cls.CnnClassifier(recognize_model, preproc_for_recognizer, model_input_shape)
    
    digit_spotter = detector.DigitSpotter(char_detector, char_recognizer, rp.MserRegionProposer())
    
    for img_file in img_files[0:]:
        # 2. image
        img = cv2.imread(img_file)

        results = [[], []]
        for i in range(2):
            if i == 0:
                image = img
            else:
                image = cv2.flip(img, -1)

            bbs, probs, y_preds = digit_spotter.run(image, threshold=0.5, do_nms=True, show_result=False, nms_threshold=0.1)
            results[i] = [bbs, probs, y_preds]
            for bb, prob, y_pred in zip(bbs, probs, y_preds):
                print img_file, i, bb, prob, y_pred

        threshold = 0.85
        if sum(prob >= threshold for prob in results[0][1]) == sum(prob >= threshold for prob in results[1][1]):
            if np.mean(results[0][1]) < np.mean(results[1][1]):
                i = 1
            else:
                i = 0
        elif sum(prob >= threshold for prob in results[0][1]) < sum(prob >= threshold for prob in results[1][1]):
            i = 1
        else:
            i = 0

        print i
        if i == 0:
            image = img
        else:
            image = cv2.flip(img, -1)

        digit_dic = {}
        bbs, probs, y_preds = results[i]
        for bb, prob, y_pred in zip(bbs, probs, y_preds):
            if prob < threshold:
                continue

            image = show.draw_box(image, bb, 2)

            y1, y2, x1, x2 = bb
            msg = "{0}".format(y_pred)
            cv2.putText(image, msg, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

            digit_dic[x1] = y_pred

        sorted_list = sorted(digit_dic.items(), key=operator.itemgetter(0))
        digit_list = [item[1] for item in sorted_list]

        print 'Predict', os.path.basename(img_file), digit_list

        cv2.imshow("MSER + CNN", image)
        # cv2.waitKey(0)

