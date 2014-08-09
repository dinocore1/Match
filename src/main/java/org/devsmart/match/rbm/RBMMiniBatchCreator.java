package org.devsmart.match.rbm;


import java.util.Collection;

public interface RBMMiniBatchCreator {

    Collection<double[]> createMiniBatch();
}
