package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;

public class DBN {

    ArrayList<RBM> rbms = new ArrayList<RBM>();

    public RealVector propagateUp(RealVector input, int numLayers) {
        for(int layer=0;layer<numLayers;layer++){
            RBM rbm = rbms.get(layer);
            input = rbm.activateHidden(input);
        }
        return input;
    }

    public RealVector propagateDown(RealVector input, int numLayers) {
        for(int layer=rbms.size()-1;layer>=0;layer--){
            RBM rbm = rbms.get(layer);
            input = rbm.activateVisible(input);
        }
        return input;
    }

}
