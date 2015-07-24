package org.devsmart.match.metric;

import org.junit.Test;
import static org.junit.Assert.*;

public class EvalTest {


    @Test
    public void testEval() {

        int[] gold = new int[] {0, 1, 2, 0, 1, 2};
        int[] pred = new int[] {0, 2, 1, 0, 0, 1};

        Evaluation<Integer> eval = new Evaluation<Integer>();

        for(int i=0;i<gold.length;i++){
            eval.update(gold[i], pred[i]);

        }

        System.out.println(eval.getSummaryTxt());


        assertEquals( 2d/6d, eval.getAccuracy(), 0.001);
        assertEquals( 0.3333, eval.getF1Score(), 0.001);



    }
}
