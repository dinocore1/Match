package org.devsmart.match.neuralnet;


import com.google.common.base.Throwables;
import org.devsmart.match.rbm.nuron.Neuron;

import java.util.ArrayList;
import java.util.LinkedList;

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

        mNeuralNet = new NeuralNet();

        mNeuralNet.inputLayer = new Neuron[mInputs.numNeurons];
        ArrayList<Neuron> lastLayer = new ArrayList<>(mInputs.numNeurons);
        for(int i=0;i<mInputs.numNeurons;i++){
            mNeuralNet.inputLayer[i] = createNeuron(mInputs.classType);
            lastLayer.add(mNeuralNet.inputLayer[i]);
        }

        mLayers.add(mOutputs);

        for(Layer layer : mLayers) {

            ArrayList<Neuron> neurons = new ArrayList<>(layer.numNeurons);
            for(int i=0;i<layer.numNeurons;i++){
                Neuron n = createNeuron(layer.classType);
                neurons.add(n);

                for(Neuron l : lastLayer) {
                    addConnection(l, n, 0);
                }
            }
            lastLayer = neurons;
        }

        mNeuralNet.outputLayer = lastLayer.toArray(new Neuron[lastLayer.size()]);

        mNeuralNet.synapses = mSynapses.toArray(new Synapse[mSynapses.size()]);

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
