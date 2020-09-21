# 
# @date: Sept 20, 2020
#
# @brief Homework 1
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
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
    
    # Test images
    img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # img_path = join(opts.data_dir, 'park/labelme_aumetbzppbkuwju.jpg')
    # img_path = join(opts.data_dir, 'laundromat/sun_aaxufyiupegixznm.jpg')
    # img_path = join(opts.data_dir, 'desert/sun_adpbjcrpyetqykvt.jpg')
    # img_path = join(opts.data_dir, 'highway/sun_beakjawckqywuhzw.jpg')
    
    ############################################################################
    # 1 Visual Words
    ############################################################################
    
    ############################################################################
    ## Q1.1.2
    # print("Q1.1.2 - Extract filter responses")
    # img = Image.open(img_path)
    # img.show()
    # img = np.array(img).astype(np.float32)/255
    # start = time()
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, filter_responses)
    # print(f"Time:  {(time() - start) / 60.0}")

    ############################################################################
    ## Q1.2
    # print("Q1.2 - Building dictionary")
    # start = time()
    # n_cpu = util.get_num_CPU()
    # visual_words.compute_dictionary(opts, n_worker=n_cpu)
    # print(f"Time: {(time() - start) / 60.0}")
    
    ############################################################################
    ## Q1.3
    # print("Q1.3 - Visual words")
    # start = time()
    # img = Image.open(img_path)
    # img.show()
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # print(f"Time: {(time() - start) / 60.0}")
    # util.visualize_wordmap(wordmap)

    ############################################################################
    # 2 Recognition System
    ############################################################################
    
    ############################################################################
    ## Q2.1 
    # print("Q2.1 - Feature from wordmap")
    # start = time()
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # hist = visual_recog.get_feature_from_wordmap(opts, wordmap)
    # print(f"Time:  {(time() - start) / 60.0}")
    # print(np.sum(hist))
    
    ############################################################################
    ## Q2.2
    # print("Q2.2 - Feature from wordmap SPM")
    # start = time()
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # hist = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    # print(f"Time:  {(time() - start) / 60.0}")
    # print(np.sum(hist))
    
    ############################################################################
    ## Q2.3 - 2.4
    # print("Q2.3-2.4 - Building recognition system")
    # start = time()
    # n_cpu = util.get_num_CPU()
    # visual_recog.build_recognition_system(opts, n_worker=n_cpu)
    # print(f"Time  {(time() - start) / 60.0}")
    
    ############################################################################
    ## Q2.5
    # print("Q2.5 - Evaluating recognition system")
    # start = time()
    # n_cpu = util.get_num_CPU()
    # conf, acc = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    # print(f"Time  {(time() - start) / 60.0}")
    # print(f"Confusion Matrix\n{conf}\n Accuracy: {acc}")
    # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [acc], fmt='%g')
    
    # hyper_params = {
    #     "filter_scales" : opts.filter_scales,
    #     "K" : opts.K,
    #     "L" : opts.L, 
    #     "alpha" : opts.alpha,
    #     "acc": acc,
    #     "conf_mat": conf 
    # }
    # np.save(join(opts.out_dir, "hyper_params.npy"), hyper_params)
    # print(np.load(join(opts.out_dir, "hyper_params.npy"), allow_pickle=True))
    
    ############################################################################
    ## Q2.6 
    print("Q2.6 Finding failures")
    conf = np.loadtxt(join(opts.out_dir, "confmat.csv"), delimiter=',')
    common_fails = visual_recog.get_common_fails(conf, opts.thresh_err)
    print(common_fails)
    preds = open(join(opts.out_dir, "pred.txt"), "r")
    for line in preds.readlines():
        pred = line.split(',')
        if pred[0] != pred[1]:
            print("Error: ", line)

    preds.close()
    
    # get confusion matrix
    # look for hard samples
    
    ############################################################################
    ## Q3.1 Hyperparameter tunning 
    # print("Q3.1 - Hyper Parameter tunning")
    # alpha = [25, 75, 125]
    # filter_scales = [[1, 2], [1, 2, 4]]
    # K = [10, 20]
    # L = [1, 2, 3]
    # tuning.tune(alpha, filter_scales, K, L)
    # print("Done")
    
    ############################################################################
    ## Q3.1 Ablation results
    # results = tuning.get_results(opts)
    # best_result = {"fs" : [0], "K": 0, "L": 0, "alpha": 0, "acc": 0}
    # for result in results:
    #     if result["acc"] > best_result["acc"]:
    #         best_result["acc"] = result["acc"]
    #         best_result["fs"] = result["filter_scales"]
    #         best_result["K"] = result["K"]
    #         best_result["L"] = result["L"]
    #         best_result["alpha"] = result["alpha"]
    # print(f"Best Model\n{best_result}")
if __name__ == '__main__':
    main()
