import os

def check_file(file):
    if os.path.isfile(file):
        return True
    else:
        print('{} not found!'.format(file))
        return False

if  (check_file('../code/LucasKanade.py') and \
     check_file('../code/LucasKanadeAffine.py') and \
     check_file('../code/SubtractDominantMotion.py') and \
     check_file('../code/InverseCompositionAffine.py') and \
     check_file('../code/testCarSequence.py') and \
     check_file('../code/testCarSequenceWithTemplateCorrection.py') and \
     check_file('../code/testGirlSequence.py') and \
     check_file('../code/testGirlSequenceWithTemplateCorrection.py') and \
     check_file('../code/testAntSequence.py') and \
     check_file('../code/testAerialSequence.py') and \
     check_file('../result/girlseqrects.npy') and \
     check_file('../result/girlseqrects-wcrt.npy') and \
     check_file('../result/carseqrects.npy') and \
     check_file('../result/carseqrects-wcrt.npy')):
    print('file check passed!')
else:
    print('file check failed!')

#modify file name according to final naming policy
#you should also include files for extra credits if you are doing them (this check file does not check for them)
#images should be be included in the report
