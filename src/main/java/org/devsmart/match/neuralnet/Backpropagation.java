package org.devsmart.match.neuralnet;


import org.devsmart.match.rbm.nuron.Neuron;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class Backpropagation {

    public ExecutorService mExecutorService;
    public NeuralNet neuralNet;
    public SMiniBatchCreator miniBatchCreator;
    public double learningRate;

    private Synapse[] weights;

    public void train(double minError, long maxEpoc) {


        {
            Collection<Synapse> synapses = neuralNet.getSynapases();
            weights = synapses.toArray(new Synapse[synapses.size())]);
        }

        for(int epoc=0;epoc<maxEpoc;epoc++){

            Collection<STE> miniBatch = miniBatchCreator.createMiniBatch();
            ArrayList<Future<double[]>> trainingTasks = new ArrayList<Future<double[]>>(miniBatch.size());
            for(STE example : miniBatch) {
                trainingTasks.add(mExecutorService.submit(new TrainingTask(example)));
            }





        }
    }

    private class TrainingTask implements Callable<double[]> {

        private final STE example;
        private NeuralNet.Context context;

        public TrainingTask(STE example) {
            this.example = example;
        }

        @Override
        public double[] call() throws Exception {
            context = neuralNet.setInputs(example.inputs[0]);

            LinkedHashSet<Neuron> next = new LinkedHashSet<Neuron>();

            for(int i=0;i<neuralNet.outputLayer.length;i++){
                Neuron n = neuralNet.outputLayer[i];
                NeuralNet.NeuronState state = context.states.get(n);
                state.error = n.derivitive(state.input) * (state.output - example.expected[i]);

                for(Synapse s : neuralNet.getInputSynapses(n)){
                    next.add(s.from);
                }
            }

            while(!next.isEmpty()){

                Iterator<Neuron> it = next.iterator();
                Neuron n = it.next();
                it.remove();

                NeuralNet.NeuronState state = context.states.get(n);

                for(Synapse s : neuralNet.getOutputSynapses(n)){
                    state.error += s.weight * context.states.get(s.to).error;
                }

                state.error = n.derivitive(state.input) * state.error;

                for(Synapse s : neuralNet.getInputSynapses(n)) {
                    next.add(s.from);
                }

            }

            double[] delta = new double[weights.size()];
            for(int i=0;i<weights.length;i++) {

                NeuralNet.NeuronState sourceState = context.states.get(weights[i].from);
                NeuralNet.NeuronState destState = context.states.get(weights[i].to);

                delta[i] = sourceState.output * destState.error;
            }

            return delta;

        }
    }
}
