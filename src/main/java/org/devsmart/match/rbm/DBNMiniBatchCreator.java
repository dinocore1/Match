package org.devsmart.match.rbm;


import java.util.Collection;

public interface DBNMiniBatchCreator {

    Collection<double[][]> createMinibatch();
}
