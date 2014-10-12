package org.devsmart.match.neuralnet;


import com.google.common.base.Throwables;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.devsmart.match.rbm.nuron.Neuron;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Backpropagation {

    public ExecutorService mExecutorService = Executors.newSingleThreadExecutor();
    public NeuralNet neuralNet;
    public MiniBatchCreator miniBatchCreator;
    public double learningRate;
    public double momentum = 0;

    public void train(double minError, long maxEpoc) {

        try {

            RealVector last = new ArrayRealVector(neuralNet.synapses.length);

            for (int epoc = 0; epoc < maxEpoc; epoc++) {

                ArrayRealVector wavg = new ArrayRealVector(neuralNet.synapses.length);

                Collection<TraningData> miniBatch = miniBatchCreator.createMiniBatch();
                ArrayList<Future<double[]>> trainingTasks = new ArrayList<Future<double[]>>(miniBatch.size());
                for (TraningData example : miniBatch) {
                    trainingTasks.add(mExecutorService.submit(new TrainingTask(example)));
                }

                for (Future<double[]> task : trainingTasks) {
                    wavg = wavg.add(new ArrayRealVector(task.get()));
                }

                wavg.mapMultiplyToSelf(1.0 / (double) miniBatch.size());

                RealVector update = wavg.add(last.mapMultiply(momentum)).mapMultiply(learningRate);
                last = update;

                //apply update
                for (int i = 0; i < update.getDimension(); i++) {
                    neuralNet.synapses[i].weight -= update.getEntry(i);
                }

            }

        } catch (Exception e) {
            Throwables.propagate(e);
        }
    }

    private class TrainingTask implements Callable<double[]> {

        private final TraningData example;
        private NeuralNet.Context context;

        public TrainingTask(TraningData example) {
            this.example = example;
        }

        @Override
        public double[] call() throws Exception {
            context = neuralNet.setInputs(example.intput);

            LinkedHashSet<Neuron> next = new LinkedHashSet<Neuron>();

            for (int i = 0; i < neuralNet.outputLayer.length; i++) {
                Neuron n = neuralNet.outputLayer[i];
                NeuralNet.NeuronState state = context.states.get(n);
                state.error = n.derivative(state.input) * (state.output - example.output[i]);

                for (Synapse s : neuralNet.getInputSynapses(n)) {
                    next.add(s.from);
                }
            }

            while (!next.isEmpty()) {

                Iterator<Neuron> it = next.iterator();
                Neuron n = it.next();
                it.remove();

                NeuralNet.NeuronState state = context.states.get(n);

                for (Synapse s : neuralNet.getOutputSynapses(n)) {
                    state.error += s.weight * context.states.get(s.to).error;
                }

                state.error = n.derivative(state.input) * state.error;

                for (Synapse s : neuralNet.getInputSynapses(n)) {
                    next.add(s.from);
                }

            }

            double[] delta = new double[neuralNet.synapses.length];
            for (int i = 0; i < delta.length; i++) {

                NeuralNet.NeuronState sourceState = context.states.get(neuralNet.synapses[i].from);
                NeuralNet.NeuronState destState = context.states.get(neuralNet.synapses[i].to);

                delta[i] = sourceState.output * destState.error;
            }

            return delta;

        }
    }
}
