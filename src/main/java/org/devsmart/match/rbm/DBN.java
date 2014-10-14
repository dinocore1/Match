package org.devsmart.match.rbm;


import org.devsmart.match.rbm.nuron.Neuron;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

public class DBN {

    public static class Layer {
        Neuron[] connected;
        Neuron[] external;
    }

    public ArrayList<Layer> mLayers;
    public ArrayList<RBM> mRBMs;

    public DBN(Collection<Layer> layers){
        mLayers = new ArrayList<Layer>(layers);
        mRBMs = new ArrayList<RBM>(mLayers.size()-1);
        for(int i=0;i<mLayers.size()-1;i++){
            Neuron[] visible = new Neuron[mLayers.get(i).connected.length + mLayers.get(i).external.length];
            System.arraycopy(mLayers.get(i).connected, 0, visible, 0, mLayers.get(i).connected.length);
            System.arraycopy(mLayers.get(i).external, 0, visible, mLayers.get(i).connected.length, mLayers.get(i).external.length);

            Neuron[] hidden = new Neuron[mLayers.get(i+1).connected.length];
            System.arraycopy(mLayers.get(i+1).connected, 0, hidden, 0, mLayers.get(i+1).connected.length);

            mRBMs.add(new RBM(visible, hidden));
        }
    }

    public double[] propagateUp(final int finalLayer, double[][] inputs, Random r) {
        int externalInput = 0;
        Layer layer = mLayers.get(0);
        double[] input = inputs[0];

        if(layer.external != null && layer.external.length > 0){
            double[] tmp = new double[layer.connected.length + layer.external.length];
            System.arraycopy(input, 0, tmp, 0, layer.connected.length);
            System.arraycopy(inputs[++externalInput], 0, tmp, layer.connected.length, layer.external.length);
            input = tmp;
        }



        for(int i=0;i<finalLayer;i++){

            input = mRBMs.get(i).activateHidden(input);

            layer = mLayers.get(i+1);
            if(layer != null && layer.external != null && layer.external.length > 0){
                double[] tmp = new double[layer.connected.length + layer.external.length];
                System.arraycopy(input, 0, tmp, 0, layer.connected.length);
                System.arraycopy(inputs[++externalInput], 0, tmp, layer.connected.length, layer.external.length);
                input = tmp;
            }
        }

        return input;
    }

}
