package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.Random;

public class DBN {

    ArrayList<RBM> rbms = new ArrayList<RBM>();

    public RealVector propagateUp(RealVector input, int numLayers, Random r) {
        for(int layer=0;layer<numLayers;layer++){
            RBM rbm = rbms.get(layer);
            input = rbm.activateHidden(input, r);
        }
        return input;
    }

    /**
     * propagate all the way down to the bottom layer
     * @param hidden nuron values
     * @param fromLayer
     * @param r
     * @return
     */
    public RealVector propagateDown(RealVector input, int fromLayer, Random r) {
        for(int layer=fromLayer;layer>=0;layer--){
            RBM rbm = rbms.get(layer);
            input = rbm.activateVisible(input, r);
        }
        return input;
    }


}
