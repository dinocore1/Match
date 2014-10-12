package org.devsmart.match.rbm;

import org.devsmart.match.rbm.nuron.SigmoidNuron;
import org.devsmart.match.rbm.nuron.Neuron;

import java.util.ArrayList;

public class DBNBuilder {

    private ArrayList<DBN.Layer> mLayers = new ArrayList<DBN.Layer>();

    public DBNBuilder addLayer(DBN.Layer layer) {
        mLayers.add(layer);
        return this;
    }

    public DBNBuilder addBernouliiLayer(final int numNurons) {
        addBernouliiLayer(numNurons, 0);
        return this;
    }

    public DBNBuilder addBernouliiLayer(final int numNurons, final int numExternal) {
        DBN.Layer layer = new DBN.Layer();
        layer.connected = new Neuron[numNurons];
        for(int i=0;i<numNurons;i++){
            layer.connected[i] = new SigmoidNuron();
        }
        layer.external = new Neuron[numExternal];
        for(int i=0;i<numExternal;i++){
            layer.external[i] = new SigmoidNuron();
        }
        addLayer(layer);
        return this;
    }

    public DBN build() {
        return new DBN(mLayers);
    }

}
