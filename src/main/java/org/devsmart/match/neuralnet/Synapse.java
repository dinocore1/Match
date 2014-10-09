package org.devsmart.match.neuralnet;


import org.devsmart.match.rbm.nuron.Neuron;

public class Synapse {

    public final Neuron from;
    public final Neuron to;
    public double weight;

    public Synapse(Neuron from, Neuron to){
        this.from = from;
        this.to = to;
    }

    @Override
    public int hashCode() {
        return from.hashCode() ^ to.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        boolean retval = false;
        if(obj instanceof Synapse){
            retval = from.equals(((Synapse) obj).from) && to.equals(((Synapse) obj).to);
        }
        return retval;
    }

    @Override
    public String toString() {
        return String.format("%s --> %s", from, to);
    }
}
