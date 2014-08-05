package org.devsmart.match.rbm;


import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MNISTDBTest {


    @Test
    public void testReadMNISTImageFile() throws Exception {
        MNISTImageFile imageFile = new MNISTImageFile(new File("build/data/train-images-idx3-ubyte"));
        BufferedImage image = imageFile.getImage(5);
        File outputFile = new File("test.png");
        ImageIO.write(image, "png", outputFile);

    }
}
