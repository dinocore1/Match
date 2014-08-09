package org.devsmart.match.rbm;


import java.util.Collection;

public interface DBNMiniBatchCreator {

    /**
     * return a new random set of training examples.
     * 1st dimention = layer
     * 2nd dimention = 0 = connected input, 1 = external input
     * 3rd dimention feature vector
     * @return
     */
    Collection<double[][][]> createMinibatch();
}
