package org.devsmart.match.neuralnet;


import java.util.Collection;

public class Backpropagation {


    public NeuralNet neuralNet;
    public SMiniBatchCreator miniBatchCreator;
    public double learningRate;

    public void train(double minError, long maxEpoc) {
        for(int epoc=0;epoc<maxEpoc;epoc++){

            Collection<STE> miniBatch = miniBatchCreator.createMiniBatch();


        }
    }
}
