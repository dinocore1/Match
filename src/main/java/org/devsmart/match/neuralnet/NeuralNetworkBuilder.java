package org.devsmart.match.neuralnet;


import com.google.common.base.Throwables;
import org.devsmart.match.rbm.nuron.Bias;
import org.devsmart.match.rbm.nuron.Neuron;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

public class NeuralNetworkBuilder {

    private class Layer {
        final int numNeurons;
        final Class<? extends Neuron> classType;

        public Layer(int numNeurons, Class<? extends Neuron> type) {
            this.numNeurons = numNeurons;
            this.classType = type;
        }
    }

    private Layer mInputs;
    private Layer mOutputs;
    private ArrayList<Layer> mLayers = new ArrayList<Layer>();

    private NeuralNet mNeuralNet;
    private LinkedList<Synapse> mSynapses = new LinkedList<Synapse>();
    public Random random = new Random();

    public NeuralNetworkBuilder withInputs(int numInputs, Class<? extends Neuron> type) {
        mInputs = new Layer(numInputs, type);
        return this;
    }

    public NeuralNetworkBuilder withOutputs(int numOutputs, Class<? extends Neuron> type) {
        mOutputs = new Layer(numOutputs, type);
        return this;
    }

    public NeuralNetworkBuilder addHiddenLayer(int numNeurons, Class<? extends Neuron> type) {
        mLayers.add(new Layer(numNeurons, type));
        return this;
    }

    public NeuralNet build() {
        if(mInputs == null){
            throw new IllegalArgumentException("must have at least one input");
        }
        if(mOutputs == null){
            throw new IllegalArgumentException("must have at least one output");
        }

        LinkedList<Neuron> allNeurons = new LinkedList<Neuron>();
        mNeuralNet = new NeuralNet();

        mNeuralNet.inputLayer = new Neuron[mInputs.numNeurons];
        ArrayList<Neuron> lastLayer = new ArrayList<>(mInputs.numNeurons);
        for(int i=0;i<mInputs.numNeurons;i++){
            mNeuralNet.inputLayer[i] = createNeuron(mInputs.classType);
            allNeurons.add(mNeuralNet.inputLayer[i]);
            lastLayer.add(mNeuralNet.inputLayer[i]);

            Bias b = new Bias();
            allNeurons.add(b);
            addConnection(b, mNeuralNet.inputLayer[i], random.nextGaussian());
        }

        mLayers.add(mOutputs);

        for(Layer layer : mLayers) {

            ArrayList<Neuron> neurons = new ArrayList<>(layer.numNeurons);
            for(int i=0;i<layer.numNeurons;i++){
                Neuron n = createNeuron(layer.classType);
                allNeurons.add(n);
                neurons.add(n);

                Bias b = new Bias();
                allNeurons.add(b);
                addConnection(b, n, 1);

                for(Neuron l : lastLayer) {
                    addConnection(l, n, random.nextGaussian());
                }
            }
            lastLayer = neurons;
        }

        mNeuralNet.outputLayer = lastLayer.toArray(new Neuron[lastLayer.size()]);

        mNeuralNet.synapses = mSynapses.toArray(new Synapse[mSynapses.size()]);
        mNeuralNet.neurons = allNeurons.toArray(new Neuron[allNeurons.size()]);

        return mNeuralNet;
    }

    private Neuron createNeuron(Class<? extends Neuron> classType) {
        try {
            return classType.newInstance();
        } catch(Throwable t) {
            Throwables.propagate(t);
            return null;
        }
    }

    private void addConnection(Neuron from, Neuron to, double weight){
        Synapse synapse = new Synapse(from, to);
        synapse.weight = weight;

        mSynapses.add(synapse);
        mNeuralNet.inputs.put(to, synapse);
        mNeuralNet.outputs.put(from, synapse);
    }
}
