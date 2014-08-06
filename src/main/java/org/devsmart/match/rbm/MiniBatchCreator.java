package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.RealVector;

import java.util.Collection;

public interface MiniBatchCreator {
    Collection<RealVector> createMiniBatch();
}
