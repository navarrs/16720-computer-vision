# 
# @date: Sept 20, 2020
#
# @brief Homework 1
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import os
from PIL import Image
from time import time

# Local includes
import util
import visual_words
import visual_recog
from opts import get_opts
import tuning

def main():
    opts = get_opts()
    
    ## Q1.1 - Filter responses
    # img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # img_path = join(opts.data_dir, 'desert/sun_aaqyzvrweabdxjzo.jpg')
    # print("Q1.1.2 - Extract filter responses")
    # img = Image.open(img_path)
    # img.show()
    # img = np.array(img).astype(np.float32)/255
    # start = time()
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # print(f"Time:  {(time() - start) / 60.0}")
    # util.display_filter_responses(opts, filter_responses)

    ## Q1.2 - Dictionary
    # print("Q1.2 - Building dictionary")
    # start = time()
    # n_cpu = util.get_num_CPU()
    # visual_words.compute_dictionary(opts, n_worker=n_cpu)
    # print(f"Time: {(time() - start) / 60.0}")
    
    ## Q1.3 - Wordmaps
    # img_path = join(opts.data_dir, 'park/labelme_aumetbzppbkuwju.jpg')
    # img_path = join(opts.data_dir, 'laundromat/sun_aaxufyiupegixznm.jpg')
    # img_path = join(opts.data_dir, 'highway/sun_beakjawckqywuhzw.jpg')
    # img_path = join(opts.data_dir, 'desert/sun_bfyksyxmxcrgvlqw.jpg')
    # img_path = join(opts.data_dir, "desert/sun_bqljzuxtzgthsjqt.jpg")
    # img_path = join(opts.data_dir, 'kitchen/sun_anmauekjnmmqhigr.jpg')
    # print("Q1.3 - Visual words")
    # start = time()
    # img = Image.open(img_path)
    # img.show()
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # print(f"Time: {(time() - start) / 60.0}")
    # util.visualize_wordmap(wordmap)

    ## Q2.1-Q2.2 - Histograms
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # print("Q2.1 - Feature from wordmap")
    # start = time()
    # hist = visual_recog.get_feature_from_wordmap(opts, wordmap)
    # print(np.sum(hist))
    # print(f"Time:  {(time() - start) / 60.0}")
    # print("Q2.2 - Feature from wordmap SPM")
    # hist = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    # print(f"Time:  {(time() - start) / 60.0}")
    # print(np.sum(hist))
    
    ## Q2.3 - 2.4 - Recognition System 
    # print("Q2.3-2.4 - Building recognition system")
    # start = time()
    # n_cpu = util.get_num_CPU()
    # visual_recog.build_recognition_system(opts, n_worker=n_cpu)
    # print(f"Time  {(time() - start) / 60.0}")
    
    ## Q2.5 - Evaluation
    # print("Q2.5 - Evaluating recognition system")
    # start = time()
    # n_cpu = util.get_num_CPU()
    # conf, acc = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    # print(f"Time  {(time() - start) / 60.0}")
    # print(f"Confusion Matrix\n{conf}\n Accuracy: {acc}")
    # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [acc], fmt='%g')
    
    ## Q2.6 - Failure Analysis
    # print("Q2.6 Finding failures") 
    # common_fails = visual_recog.get_common_fails(opts)
    # print(common_fails)
    
    ## Q3.1 - Hyperparameter tunning 
    # print("Q3.1 - Hyper Parameter tunning")
    # alpha = [25, 125]
    # filter_scales = [[1, 2], [1, 2, 4]]
    # K = [10, 30, 50]
    # L = [3, 2, 1]
    # tuning.tune(alpha, filter_scales, K, L)
    # print("Done")
    
    ## Q3.1 - Ablation results    
    results = tuning.get_results(opts, sorted=True)
    tuning.display_results(results)
    
if __name__ == '__main__':
    main()
