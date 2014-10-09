package org.devsmart.match.neuralnet;


import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import org.devsmart.match.rbm.nuron.Neuron;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public class NeuralNet {

    public class Context {

        class NeuronState {
            final Neuron neuron;
            double input = 0;
            double output = 0;

            public NeuronState(Neuron neuron) {
                this.neuron = neuron;
            }
        }

        HashMap<Neuron, NeuronState> states = new HashMap<Neuron, NeuronState>();

        public Context(double[] inputs) {
            Set<Neuron> next = new HashSet<Neuron>();
            for(int i=0;i<inputLayer.length;i++) {
                NeuronState state = new NeuronState(inputLayer[i]);
                state.output = inputs[i];
                states.put(inputLayer[i], state);

                for(Synapse s : outputs.get(inputLayer[i])){
                    next.add(s.to);
                }
            }

            while(!next.isEmpty()){
                next = feedForward(next);
            }

        }

        private Set<Neuron> feedForward(Iterable<Neuron> neurons) {
            Set<Neuron> next = new HashSet<>();
            for(Neuron n : neurons) {
                NeuronState state = states.get(n);
                if(state == null) {
                    state = new NeuronState(n);
                    states.put(n, state);
                }
                assert state.input == 0;
                for(Synapse s : inputs.get(n)){
                    state.input += states.get(s.from).output * s.weight;
                }
                state.output = n.value(state.input);

                for(Synapse s : outputs.get(n)){
                    next.add(s.to);
                }
            }
            return next;
        }

        public double[] getOutputs() {
            double[] retval = new double[outputLayer.length];
            for(int i=0;i<retval.length;i++) {
                NeuronState s = states.get(outputLayer[i]);
                retval[i] = s.output;
            }
            return retval;
        }


    }

    Neuron[] inputLayer;
    Neuron[] outputLayer;

    private Set<Neuron> neurons = new HashSet<Neuron>();
    private ListMultimap<Neuron, Synapse> inputs = LinkedListMultimap.create();
    private ListMultimap<Neuron, Synapse> outputs = LinkedListMultimap.create();


    public void addConnection(Neuron from, Neuron to, double weight){
        neurons.add(from);
        neurons.add(to);
        Synapse synapse = new Synapse(from, to);
        synapse.weight = weight;
        inputs.put(to, synapse);
        outputs.put(from, synapse);
    }

    public Collection<Synapse> getInputSynapses(Neuron neuron) {
        return inputs.get(neuron);
    }

    public Collection<Synapse> getOutputSynapses(Neuron neuron) {
        return outputs.get(neuron);
    }

    public Context setInputs(double[] inputs) {
        if(inputs == null || inputs.length != inputLayer.length){
            throw new IllegalArgumentException("number of inputs must match input layer");
        }
        return new Context(inputs);
    }


}
