package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.devsmart.match.rbm.nuron.Nuron;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

public class DBN {

    public static class Layer {
        Nuron[] connected;
        Nuron[] external;
    }

    ArrayList<Layer> mLayers;
    ArrayList<RBM> mRBMs;

    public DBN(Collection<Layer> layers){
        mLayers = new ArrayList<>(layers);
        mRBMs = new ArrayList<RBM>(mLayers.size()-1);
        for(int i=0;i<mLayers.size()-1;i++){
            Nuron[] hidden = new Nuron[mLayers.get(i+1).connected.length + mLayers.get(i+1).external.length];
            System.arraycopy(mLayers.get(i+1).connected, 0, hidden, 0, mLayers.get(i+1).connected.length);
            System.arraycopy(mLayers.get(i+1).external, 0, hidden, mLayers.get(i+1).connected.length, mLayers.get(i+1).external.length);
            mRBMs.add(new RBM(mLayers.get(i).connected, hidden));
        }
    }

    public RealVector propagateUp(int finalLayer, double[][] input, Random r) {
        RealVector hidden;
        for(int layer=0;layer<finalLayer;layer++){
            RBM rbm = mRBMs.get(layer);

            double[] visible = new double[rbm.visible.length];
            for(int i=0;i<visible.length;i++){
                visible[i] = input[layer][i];
            }
            hidden = rbm.activateHidden(new ArrayRealVector(visible), r);
        }

        return hidden;
    }

}
