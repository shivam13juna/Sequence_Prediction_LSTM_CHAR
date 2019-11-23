

import tensorflow as tf
import numpy as np
import my_txtutils

# these must match what was saved !
ALPHASIZE = my_txtutils.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512


davinciC0 = "checkpoints/rnn_train_1495455686-0"      # random
davinciC1 = "checkpoints/rnn_train_1495455686-150000"  # lower case gibberish
davinciC2 = "checkpoints/rnn_train_1495455686-300000"  # words, paragraphs
davinciC3 = "checkpoints/rnn_train_1495455686-450000"  # structure of a play, unintelligible words
davinciC4 = "checkpoints/rnn_train_1495447371-15000000"  # better structure of a play, character names (not very good), 4-letter words in correct English
davinciC5 = "checkpoints/rnn_train_1495447371-45000000"  # good names, even when invented (ex: SIR NATHANIS LORD OF SYRACUSE), correct 6-8 letter words
davinciB10 = "checkpoints/rnn_train_1495440473-102000000" # ACT V SCENE IV, [Re-enter KING JOHN with MARDIAN], DON ADRIANO DRAGHAMONE <- invented!
# most scene directions correct: [Enter FERDINAND] [Dies] [Exit ROSALIND] [To COMINIUS with me] [Enter PRINCE HENRY, and Attendants], correct English.

pythonA0 = "checkpoints/rnn_train_1495458538-300000"  # gibberish
pythonA1 = "checkpoints/rnn_train_1495458538-1200000"  # some function calls with parameters and ()
pythonA2 = "checkpoints/rnn_train_1495458538-10200000"  # starts looking Tensorflow Python, nested () and [] not perfect yet
pythonB10 = "checkpoints/rnn_train_1495458538-201600000"  # can even recite the Apache license

# use topn=10 for all but the last one which works with topn=2 for davinci and topn=3 for Python
author = davinciB10

ncnt = 0
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('checkpoints/rnn_train_1495455686-0.meta')
    new_saver.restore(sess, author)
    x = my_txtutils.convert_from_alphabet(ord("L"))
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    for i in range(1000000000):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        # If sampling is be done from the topn most likely characters, the generated text
        # is more credible and more "english". If topn is not set, it defaults to the full
        # distribution (ALPHASIZE)

        # Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

        c = my_txtutils.sample_from_probabilities(yo, topn=2)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = chr(my_txtutils.convert_to_alphabet(c))
        print(c, end="")

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 100:
            print("")
            ncnt = 0

