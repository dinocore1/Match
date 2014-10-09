package org.devsmart.match.rbm.nuron;


import java.util.HashSet;

public abstract class Neuron {

    public abstract double value(double input);
    public abstract double derivitive(double input);

    public final HashSet<Neuron> inputs = new HashSet<Neuron>();
    public final HashSet<Neuron> output = new HashSet<Neuron>();

}
