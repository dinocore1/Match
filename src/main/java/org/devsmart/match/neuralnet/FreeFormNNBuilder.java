package org.devsmart.match.neuralnet;


import com.google.common.base.Throwables;

import org.devsmart.match.rbm.nuron.Bias;
import org.devsmart.match.rbm.nuron.Neuron;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

public class FreeFormNNBuilder {

    private NeuralNet mNeuralNet = new NeuralNet();
    private ArrayList<Layer> mLayers = new ArrayList<Layer>();
    private LinkedList<Synapse> mSynapses = new LinkedList<Synapse>();

    public Random random = new Random();

    public Neuron[] createLayer(int numNeurons, Class<? extends Neuron> classType) {
        Layer l = new Layer();
        l.neurons = new Neuron[numNeurons];
        for(int i=0;i<numNeurons;i++){
            l.neurons[i] = createNeuron(classType);
        }
        mLayers.add(l);
        return l.neurons;
    }


    private static Neuron createNeuron(Class<? extends Neuron> classType) {
        try {
            return classType.newInstance();
        } catch(Throwable t) {
            Throwables.propagate(t);
            return null;
        }
    }

    public void addBias(Neuron[] neurons){
        for(Neuron n : neurons) {
            addBias(n);
        }
    }

    public void addBias(Neuron n) {
        Bias b = new Bias();
        addConnection(b, n, 1);
    }

    public void addConnection(Neuron from, Neuron to, double weight){
        Synapse synapse = new Synapse(from, to);
        synapse.weight = weight;

        mSynapses.add(synapse);
        mNeuralNet.inputs.put(to, synapse);
        mNeuralNet.outputs.put(from, synapse);
    }

    public void fullyConnect(Neuron[] from, Neuron[] to) {
        for(int i=0;i<from.length;i++){
            for(int j=0;j<to.length;j++){
                addConnection(from[i], to[j], random.nextGaussian());
            }
        }
    }

    public NeuralNet build() {
        mNeuralNet.layers = mLayers.toArray(new Layer[mLayers.size()]);
        mNeuralNet.synapses = mSynapses.toArray(new Synapse[mSynapses.size()]);
        return mNeuralNet;
    }
}
