package org.devsmart.match.neuralnet;


import com.amd.aparapi.Kernel;
import com.amd.aparapi.ProfileInfo;
import com.amd.aparapi.Range;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Test;

import java.util.List;
import java.util.Random;

public class KernelTest {


    static class SquaresKernel extends Kernel {

        public final float[] input;
        public final float[] output;

        public SquaresKernel(int size) {
            input = new float[size];
            output = new float[size];
        }

        @Override public void run() {
            int gid = getGlobalId();
            output[gid] = input[gid] * input[gid];
        }
    }

    @Test
    public void testSquares() {
        SummaryStatistics jtpStats = new SummaryStatistics();
        SummaryStatistics gpuStats = new SummaryStatistics();
        Random r = new Random(1);

        final int numInputs = 40000;


        {
            SquaresKernel kernel = new SquaresKernel(numInputs);

            for (int i = 0; i < kernel.input.length; i++) {
                kernel.input[i] = r.nextFloat();
            }


            kernel.setExecutionMode(Kernel.EXECUTION_MODE.GPU);

            kernel.setExplicit(true);
            kernel.put(kernel.input);
            for (int i = 0; i < 1000; i++) {
                kernel.execute(numInputs);
                //sigmoidKernel.get(sigmoidKernel.outputValues);
                long executionTime = kernel.getExecutionTime();
                gpuStats.addValue(executionTime);
            }
        }

        {
            SquaresKernel kernel = new SquaresKernel(numInputs);

            for (int i = 0; i < kernel.input.length; i++) {
                kernel.input[i] = r.nextFloat();
            }


            kernel.setExecutionMode(Kernel.EXECUTION_MODE.CPU);

            kernel.setExplicit(true);
            kernel.put(kernel.input);
            for (int i = 0; i < 1000; i++) {
                kernel.execute(numInputs);
                //sigmoidKernel.get(sigmoidKernel.outputValues);
                long executionTime = kernel.getExecutionTime();
                jtpStats.addValue(executionTime);
            }
        }



        System.out.println(String.format("squares jtp: %g gpu: %g", jtpStats.getMean(), gpuStats.getMean()));
    }


    @Test
    public void testRBMKernel() {

        SummaryStatistics jtpStats = new SummaryStatistics();
        SummaryStatistics gpuStats = new SummaryStatistics();
        Random r = new Random(1);

        final int numInputs = 40000;
        final int numOutputs = 500;



        {
            SigmoidKernel sigmoidKernel = new SigmoidKernel(numInputs, numOutputs);

            for (int i = 0; i < sigmoidKernel.inputValues.length; i++) {
                sigmoidKernel.inputValues[i] = r.nextFloat();
            }

            for (int i = 0; i < sigmoidKernel.weights.length; i++) {
                sigmoidKernel.weights[i] = r.nextFloat();
            }


            sigmoidKernel.setExecutionMode(Kernel.EXECUTION_MODE.GPU);

            sigmoidKernel.setExplicit(true);
            sigmoidKernel.put(sigmoidKernel.inputValues);
            sigmoidKernel.put(sigmoidKernel.weights);
            for (int i = 0; i < 1000; i++) {
                sigmoidKernel.execute(numOutputs);
                //sigmoidKernel.get(sigmoidKernel.outputValues);
                long executionTime = sigmoidKernel.getExecutionTime();
                gpuStats.addValue(executionTime);
            }
        }

        {
            SigmoidKernel sigmoidKernel = new SigmoidKernel(numInputs, numOutputs);

            for (int i = 0; i < sigmoidKernel.inputValues.length; i++) {
                sigmoidKernel.inputValues[i] = r.nextFloat();
            }

            for (int i = 0; i < sigmoidKernel.weights.length; i++) {
                sigmoidKernel.weights[i] = r.nextFloat();
            }


            sigmoidKernel.setExecutionMode(Kernel.EXECUTION_MODE.CPU);

            sigmoidKernel.setExplicit(true);
            sigmoidKernel.put(sigmoidKernel.inputValues);
            sigmoidKernel.put(sigmoidKernel.weights);
            for (int i = 0; i < 1000; i++) {
                sigmoidKernel.execute(numOutputs);
                //sigmoidKernel.get(sigmoidKernel.outputValues);
                long executionTime = sigmoidKernel.getExecutionTime();
                jtpStats.addValue(executionTime);
            }
        }



        System.out.println(String.format("jtp: %g gpu: %g", jtpStats.getMean(), gpuStats.getMean()));



    }
}
