package org.devsmart.match.neuralnet;


import java.util.Collection;

public interface MiniBatchCreator {

    Collection<TraningData> createMiniBatch();
}
