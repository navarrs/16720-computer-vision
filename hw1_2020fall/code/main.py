from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts


def main():
    opts = get_opts()

    ## Q1.1
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    img = Image.open(img_path)
    img.show()
    img = np.array(img).astype(np.float32)/255
    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses)

    ## Q1.2
    # print("Building dictionary")
    # n_cpu = util.get_num_CPU()
    # visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    ## Q1.3
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # util.visualize_wordmap(wordmap)

    ## Q2.1-2.4
    # hist = visual_recog.get_feature_from_wordmap(opts, wordmap)
    # hist = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    # hist = visual_recog.get_image_feature(opts, img_path, dictionary)
    # print(hist, np.sum(hist))
    # print("Building recognition system")
    # n_cpu = util.get_num_CPU()
    # visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    ## Q2.5
    # print("Evaluating recognition system")
    # n_cpu = util.get_num_CPU()
    # conf, accuracy = visual_recog.evaluate_recognition_system(opts, 
    #                                                           n_worker=n_cpu)
    # print(conf)
    # print(accuracy)
    # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')
    
    # hyper_params = {
    #     "filter_scales" : opts.filter_scales,
    #     "K" : opts.K,
    #     "L" : opts.L, 
    #     "alpha" : opts.alpha 
    # }
    # np.save(join(opts.out_dir, "hyper_params.npy"), hyper_params)
    # print(np.load(join(opts.out_dir, "hyper_params.npy"), allow_pickle=True))
    
if __name__ == '__main__':
    main()
