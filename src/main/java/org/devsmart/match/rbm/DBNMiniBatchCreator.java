package org.devsmart.match.rbm;


import java.util.Collection;

public interface DBNMiniBatchCreator {

    /**
     * return a new random set of training examples.
     * 1st dimention = input set. 0=bottom visible, 1>= extra inputs
     * 2nd dimention = data vector
     * @return
     */
    Collection<double[][]> createMinibatch();
}
