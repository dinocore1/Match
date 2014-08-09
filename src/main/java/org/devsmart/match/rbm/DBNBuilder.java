package org.devsmart.match.rbm;

import org.devsmart.match.rbm.nuron.BernoulliNuron;
import org.devsmart.match.rbm.nuron.Nuron;

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
        layer.connected = new Nuron[numNurons];
        for(int i=0;i<numNurons;i++){
            layer.connected[i] = new BernoulliNuron();
        }
        layer.external = new Nuron[numExternal];
        for(int i=0;i<numExternal;i++){
            layer.external[i] = new BernoulliNuron();
        }
        addLayer(layer);
        return this;
    }

    public DBN build() {
        return new DBN(mLayers);
    }

}
