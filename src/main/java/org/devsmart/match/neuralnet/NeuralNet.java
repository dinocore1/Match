package org.devsmart.match.neuralnet;


import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import org.devsmart.match.rbm.nuron.Bias;
import org.devsmart.match.rbm.nuron.Neuron;

import java.util.*;

public class NeuralNet {

    public static class NeuronState {
        final Neuron neuron;
        double input = 0;
        double output = 0;
        double error = 0;

        public NeuronState(Neuron neuron) {
            this.neuron = neuron;
        }
    }

    public class Context {

        HashMap<Neuron, NeuronState> states = new HashMap<Neuron, NeuronState>();

        public Context(double[] inputsValues) {

            HashSet<Neuron> neurons = new HashSet<Neuron>();
            for(Synapse s : synapses){
                neurons.add(s.from);
                neurons.add(s.to);
            }

            for(Neuron n : neurons){
                NeuronState s = new NeuronState(n);
                if(n instanceof Bias){
                    s.output = 1.0;
                }
                states.put(n, s);
            }

            for(int i=0;i<layers[0].neurons.length;i++){
                Neuron n = layers[0].neurons[i];
                NeuronState state = states.get(n);
                state.output = inputsValues[i];
            }

            for(int l=1;l<layers.length;l++){
                for(Neuron n : layers[l].neurons){
                    NeuronState state = states.get(n);

                    for(Synapse s : inputs.get(n)){
                        state.input += states.get(s.from).output * s.weight;
                    }

                    state.output = n.value(state.input);

                }
            }




        }

        public double[] getOutputs() {
            double[] retval = new double[layers[layers.length-1].neurons.length];
            for(int i=0;i<retval.length;i++) {
                NeuronState s = states.get(layers[layers.length-1].neurons[i]);
                retval[i] = s.output;
            }
            return retval;
        }


    }


    Layer[] layers;
    Synapse[] synapses;

    ListMultimap<Neuron, Synapse> inputs = LinkedListMultimap.create();
    ListMultimap<Neuron, Synapse> outputs = LinkedListMultimap.create();



    public Collection<Synapse> getInputSynapses(Neuron neuron) {
        return inputs.get(neuron);
    }

    public Collection<Synapse> getOutputSynapses(Neuron neuron) {
        return outputs.get(neuron);
    }

    public Context setInputs(double[] inputs) {
        if(inputs == null || inputs.length != layers[0].neurons.length){
            throw new IllegalArgumentException("number of inputs must match input layer");
        }
        return new Context(inputs);
    }

    public Neuron[] getOutputLayer() {
        return layers[layers.length-1].neurons;
    }


}
