package org.devsmart.match.rbm;


import org.junit.Test;
import static org.junit.Assert.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MNISTDBTest {


    @Test
    public void testReadMNISTImageFile() throws Exception {
        MNISTImageFile imageFile = new MNISTImageFile(new File("build/data/train-images-idx3-ubyte"));
        double[] data = imageFile.getImage(5);
        assertEquals(0.1568627450980392, data[207], 0.0001);
        assertEquals(0.9882352941176471, data[210], 0.0001);


        BufferedImage image = imageFile.getBufferedImage(5);
        File outputFile = new File("test.png");
        ImageIO.write(image, "png", outputFile);

    }
}
