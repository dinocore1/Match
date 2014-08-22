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
        mLayers = new ArrayList<Layer>(layers);
        mRBMs = new ArrayList<RBM>(mLayers.size()-1);
        for(int i=0;i<mLayers.size()-1;i++){
            Nuron[] visible = new Nuron[mLayers.get(i).connected.length + mLayers.get(i).external.length];
            System.arraycopy(mLayers.get(i).connected, 0, visible, 0, mLayers.get(i).connected.length);
            System.arraycopy(mLayers.get(i).external, 0, visible, mLayers.get(i).connected.length, mLayers.get(i).external.length);

            Nuron[] hidden = new Nuron[mLayers.get(i+1).connected.length];
            System.arraycopy(mLayers.get(i+1).connected, 0, hidden, 0, mLayers.get(i+1).connected.length);

            mRBMs.add(new RBM(visible, hidden));
        }
    }

    public double[] getVisibleForLayer(int layer, double[][][] input, double[] lastVisible) {
        final RBM rbm = mRBMs.get(layer);
        final double[][] layerInput = input[layer];

        if(lastVisible == null){
            lastVisible = new double[rbm.visible.length];
        }

        if(lastVisible.length != rbm.visible.length){
            double[] tmp = new double[rbm.visible.length];
            System.arraycopy(lastVisible, 0, tmp, 0, lastVisible.length);
            lastVisible = tmp;
        }
        if(layerInput[0] != null){
            System.arraycopy(layerInput[0], 0, lastVisible, 0, layerInput[0].length);
        }
        if(layerInput[1] != null){
            System.arraycopy(layerInput[1], 0, lastVisible, lastVisible.length-layerInput[1].length, layerInput[1].length);
        }

        return lastVisible;
    }

    public double[] propagateUp(int finalLayer, double[][][] input, Random r) {
        double[] visible = null;
        for(int i=0;i<finalLayer;i++){
            RBM rbm = mRBMs.get(i);
            visible = getVisibleForLayer(i, input, visible);
            visible = rbm.getHiddenInput(visible);
        }

        return visible;
    }

}
