package org.devsmart.match;


import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.junit.Test;
import static org.junit.Assert.*;

public class PCATest {

    //first dimension refers to the data rows
    private static final double[][] RAW_DATA = new double[][]{
            {2.5, 2.4},
            {0.5, 0.7},
            {2.2, 2.9},
            {1.9, 2.2},
            {3.1, 3.0},
            {2.3, 2.7},
            {2.0, 1.6},
            {1.0, 1.1},
            {1.5, 1.6},
            {1.1, 0.9}
    };

    private static final double[] expectedFirstPCAResult = new double[]{
            -0.827970186,
            1.77758033,
            -0.992197494,
            -0.274210416,
            -1.67580142,
            -.912949103,
            0.0991094375,
            1.14457216,
            0.438046137,
            1.22382056
    };

    /**
     * the sign of the PC is not important as long as it is consistant.
     * That is why the abs value is taken in the assertEquals below
     */

    @Test
    public void testSVD_PCA() {
        RealMatrix standardizedData = DataUtils.centerMean(RAW_DATA, true);
        SingularValueDecomposition decomposition = new SingularValueDecomposition(standardizedData);
        RealMatrix pcaResults = decomposition.getU().multiply(decomposition.getS());

        for(int i=0;i<10;i++){
            assertEquals(Math.abs(expectedFirstPCAResult[i]),
                    Math.abs(pcaResults.getEntry(i, 0)),
                    0.01);
        }
    }

    @Test
    public void testPCA() throws Exception {

        PCA pca = new PCA();
        pca.computeTransform(RAW_DATA, true, false);

        for(int i=0;i<10;i++){
            RealMatrix result = pca.getPrincipalComponents(RAW_DATA[i], false);
            assertEquals(Math.abs(expectedFirstPCAResult[i]),
                    Math.abs(result.getEntry(0, 0)),
                    0.01);

        }


    }
}
