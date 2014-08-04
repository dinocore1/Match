package org.devsmart.match.rbm;

import java.util.ArrayList;
import java.util.Random;

public class DBNBuilder {

    private ArrayList<Integer> mLayerNumInput = new ArrayList<Integer>();
    private ArrayList<Boolean> mLayerIsGaussian = new ArrayList<Boolean>();
    private Random r = new Random();


    public DBNBuilder addLayer(int numInputs, boolean isGaussian) {
        mLayerNumInput.add(numInputs);
        mLayerIsGaussian.add(isGaussian);
        return this;
    }

    public DBNBuilder withRandom(Random r) {
        this.r = r;
        return this;
    }

    public DBN build() {
        DBN retval = new DBN();
        for(int i=1;i<mLayerNumInput.size();i++){
            RBM rbm = new RBM(mLayerNumInput.get(i-1), mLayerNumInput.get(i), mLayerIsGaussian.get(i-1), r);
            retval.rbms.add(rbm);
        }


        return retval;
    }

}
