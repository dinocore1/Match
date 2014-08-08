package org.devsmart.match.rbm;

import org.devsmart.match.rbm.nuron.BernoulliNuron;
import org.devsmart.match.rbm.nuron.GaussianNuron;
import org.devsmart.match.rbm.nuron.Nuron;

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

    private Nuron[] createLayer(int size, Class<?> nuronClass) throws IllegalAccessException, InstantiationException {
        Nuron[] retval = new Nuron[size];
        for(int i=0;i<size;i++){
            retval[i] = (Nuron)nuronClass.newInstance();
        }
        return retval;
    }

    public DBN build() {
        try {
            DBN retval = new DBN();
            for (int i = 1; i < mLayerNumInput.size(); i++) {
                Nuron[] visible = createLayer(mLayerNumInput.get(i - 1), mLayerIsGaussian.get(i - 1) ? GaussianNuron.class : BernoulliNuron.class);
                Nuron[] hidden = createLayer(mLayerNumInput.get(i), mLayerIsGaussian.get(i) ? GaussianNuron.class : BernoulliNuron.class);
                RBM rbm = new RBM(visible, hidden);
                retval.rbms.add(rbm);
            }
            return retval;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}
